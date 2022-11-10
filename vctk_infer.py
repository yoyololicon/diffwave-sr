import argparse
from copy import deepcopy
from typing import Union
import math
import os
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
import torchaudio
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor, nn
from torch.cuda import amp
from tqdm import tqdm
from typing import Callable, List, Tuple
from multiprocessing import Process, Queue, set_start_method
from functools import partial
from kazane import Decimate, Upsample
from samplerate import resample
from scipy.signal import cheby1

from utils import gamma2snr, snr2as, gamma2as, gamma2logas, get_instance
import models as module_arch

SAMPLERATE = 48000
sinc_kwargs = {
    'roll_off': 0.962,
    'num_zeros': 128,
    'window_func': partial(torch.kaiser_window, periodic=False,
                           beta=14.769656459379492),
}


class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    # x: [B,T], r: [B], int
    def forward(self, x, r):
        origin_shape = x.shape
        T = origin_shape[-1]
        x = x.view(-1, T)

        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )  # return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2], 1, 1)
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )  # return_complex=False)
        x = x[:, :T]
        return x.view(*origin_shape)


class STFTDecimate(LowPass):
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, ratio=[1 / r], **kwargs)
        self.r = r

    def forward(self, x):
        return super().forward(x, 0)[..., ::self.r]


class ChebyDecimate(nn.Module):
    def __init__(self, r, *args, **kwargs):
        super().__init__()
        self.r = r
        self.sos = cheby1(8, 0.05, 1 / r, 'lowpass', output='sos')

    def forward(self, x):
        device = x.device
        x = x.cpu()
        for i in range(self.sos.shape[0]):
            x = torchaudio.functional.filtering.biquad(x, *self.sos[i])
        return x[..., ::self.r].to(device)


class ChebyUpsample(nn.Module):
    def __init__(self, r, *args, **kwargs):
        super().__init__()
        self.r = r
        self.sos = cheby1(8, 0.05, 1 / r, 'lowpass', output='sos')

    def forward(self, x):
        *shape, _ = x.shape
        x = F.upsample(x.view(-1, 1, x.size(-1)),
                       scale_factor=self.r, mode='nearest')
        device = x.device
        x = x.cpu().flip(-1)
        for i in range(self.sos.shape[0]):
            x = torchaudio.functional.filtering.biquad(x, *self.sos[i])
        return x.flip(-1).to(device).view(*shape, -1)


class LSD(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, y_hat, y, ratio=None):
        Y_hat = torch.stft(y_hat, self.n_fft, hop_length=self.hop_length,
                           window=self.window, return_complex=True)
        Y = torch.stft(y, self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        if ratio is not None:
            index = int((self.n_fft // 2 + 1) * ratio)
            Y_hat = Y_hat[:index]
            Y = Y[:index]
        sp = Y_hat.abs().square_().clamp_(min=1e-8).log10_()
        st = Y.abs().square_().clamp_(min=1e-8).log10_()
        return (sp - st).square_().mean(0).sqrt_().mean()


@torch.no_grad()
def reverse(y_hat,
            gamma,
            downsample: Callable,
            upsample: Callable,
            inference_func: Callable,
            verbose=True):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    var_st = torch.exp(log_var[:-1] - log_var[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    def degradation_func(x): return upsample(downsample(x))

    z_t = torch.randn_like(y_hat)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_func(z_t, t)

        # clip noise
        # noise_hat = noise_hat.clamp_(
        #    (z_t - alpha[t]) * var[t].rsqrt(),
        #    (alpha[t] + z_t) * var[t].rsqrt(),
        # )

        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]

        mu = mu - degradation_func(mu)
        mu += degradation_func(z_t) * \
            var_st[s] / alpha_st[s] + alpha[s] * c[s] * y_hat

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    noise_hat = inference_func(z_t, 0)
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
    return final


@torch.no_grad()
def nuwave_reverse(y_hat, gamma, inference_model: Callable, verbose=True):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    z_t = torch.randn_like(y_hat)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_model(z_t, y_hat, alpha[t:t+1])
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    noise_hat = inference_model(z_t, y_hat, alpha[:1])
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
    return final


@torch.no_grad()
def nuwave2_reverse(y_hat, band, gamma, inference_model: Callable, verbose=True):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    norm_nlogsnr = (gamma + 20) / 40
    norm_nlogsnr.clip_(0, 1)
    z_t = torch.randn_like(y_hat)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_model(z_t, y_hat, band, norm_nlogsnr[t:t+1])
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    noise_hat = inference_model(z_t, y_hat, band, norm_nlogsnr[:1])
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
    return final


@torch.no_grad()
def nuwave2_ddim(y_hat, band, gamma, inference_model: Callable, verbose=True):
    Pm1 = -torch.expm1((gamma[1:] - gamma[:-1]) * 0.5)
    log_alpha, log_var = gamma2logas(gamma)
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    std = log_var.mul(0.5).exp()
    alpha = log_alpha.exp()
    T = gamma.numel()

    norm_nlogsnr = (gamma + 20) / 40
    norm_nlogsnr.clip_(0, 1)
    z_t = torch.randn_like(y_hat)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_model(z_t, y_hat, band, norm_nlogsnr[t:t+1])
        z_t.mul_(alpha_st[s]).add_(std[s] * Pm1[s] * noise_hat)

    noise_hat = inference_model(z_t, y_hat, band, norm_nlogsnr[:1])
    final = (z_t - std[0] * noise_hat) / log_alpha[0].exp()
    return final


def reverse_manifold(y_hat,
                     gamma,
                     downsample: Callable,
                     upsample: Callable,
                     inference_func: Callable,
                     lr: float = 1,
                     verbose=True,
                     inter_results=False):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    var_st = torch.exp(log_var[:-1] - log_var[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    def degradation_func(x): return upsample(downsample(x))

    z_t = torch.randn_like(y_hat)

    window_size = 144000 // 2
    overlap = 12000 // 2
    hop_size = window_size - overlap
    p = torch.linspace(0, 1, overlap, device=z_t.device)

    middle_results = []

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = torch.zeros_like(z_t)
        correction_grad = torch.zeros_like(z_t)
        for i in range(0, z_t.shape[1] - overlap, hop_size):
            indexes = slice(i, i + window_size)
            sub_z_t = z_t[:, indexes].clone().requires_grad_(True)
            sub_noise_hat = inference_func(sub_z_t, t, indexes)
            x_hat = (sub_z_t - var[t].sqrt() * sub_noise_hat) / alpha[t]
            x_hat.clamp_(-1, 1)
            loss = F.mse_loss(degradation_func(x_hat),
                              y_hat[:, indexes], reduction='sum')
            loss = loss / sub_noise_hat.numel()
            g, *_ = grad(loss, sub_z_t)
            torch.nan_to_num_(g, nan=0)
            g = g * sub_noise_hat.numel()
            assert not torch.isnan(g).any(), 'NaN gradient'
            sub_noise_hat = sub_noise_hat.detach()
            if i > 0:
                noise_hat[:, i:i+overlap] *= 1 - p
                correction_grad[:, i:i+overlap] *= 1 - p
                sub_noise_hat[:, :overlap] *= p
                g[:, :overlap] *= p

            noise_hat[:, indexes] += sub_noise_hat
            correction_grad[:, indexes] += g

        assert not torch.isnan(noise_hat).any(), 'NaN noise_hat'
        assert not torch.isnan(correction_grad).any(), 'NaN correction_grad'

        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]
        mu -= correction_grad * lr
        mu = mu - degradation_func(mu)

        if inter_results:
            x_hat = (z_t - var[t].sqrt() * noise_hat) / alpha[t]
            middle_results.append(x_hat - degradation_func(x_hat) + y_hat)

        mu += degradation_func(z_t) * \
            var_st[s] / alpha_st[s] + alpha[s] * c[s] * y_hat

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    with torch.no_grad():
        noise_hat = inference_func(z_t, 0, slice(None))
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]

    if inter_results:
        return final, middle_results
    return final


def foo(fq: Queue, rq: Queue, q: int, infer_type: str, lr: float, target_sr: Union[int, None],
        model, evaluater, downsampler, upsampler, gamma, steps):
    try:
        alpha = gamma2as(gamma)[0]
        while not fq.empty():
            filename = fq.get()
            device = gamma.device

            raw_y, sr = torchaudio.load(filename)
            raw_y = raw_y.to(device)

            offset = raw_y.shape[1] % q
            if offset:
                y = raw_y[:, :-offset]
            else:
                y = raw_y

            y_lowpass = downsampler(y)

            if infer_type == "nuwave":
                y_hat = F.upsample(y_lowpass.unsqueeze(
                    1), scale_factor=q, mode='linear', align_corners=False).squeeze(1)
                y_recon = nuwave_reverse(y_hat, gamma,
                                         amp.autocast()(model),
                                         verbose=False)
            elif infer_type == "inpainting":
                y_hat = upsampler(y_lowpass)
                y_recon = reverse(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t: model(
                        x, steps[t:t+1])),
                    verbose=False
                )
            elif infer_type == "nuwave-inpainting":
                y_hat = upsampler(y_lowpass)
                nuwave_cond = F.upsample(y_lowpass.unsqueeze(
                    1), scale_factor=q, mode='linear', align_corners=False).squeeze(1)
                y_recon = reverse(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t: model(
                        x, nuwave_cond, alpha[t:t+1])),
                    verbose=False
                )
            elif infer_type == "nuwave-manifold":
                y_hat = upsampler(y_lowpass)
                nuwave_cond = F.upsample(y_lowpass.unsqueeze(
                    1), scale_factor=q, mode='linear', align_corners=False).squeeze(1)
                y_recon = reverse_manifold(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t, idx: model(
                        x, nuwave_cond[:, idx], alpha[t:t+1])),
                    lr=lr,
                    verbose=False
                )
            elif infer_type == "manifold":
                y_hat = upsampler(y_lowpass)
                y_recon = reverse_manifold(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast(enabled=False)(lambda x, t, idx: model(
                        x, steps[t:t+1])),
                    lr=lr,
                    verbose=False
                )
            elif infer_type == "nuwave2":
                scaler = y.abs().max()
                y_hat = upsampler(y_lowpass) / scaler
                band = y_hat.new_zeros((1, 513), dtype=torch.int64)
                band[:, :int(513 / q)] = 1
                y_recon = nuwave2_reverse(y_hat, band, gamma - 2 * scaler.log(),
                                          amp.autocast()(model),
                                          verbose=False) * scaler
            elif infer_type == "nuwave2-manifold":
                scaler = y.abs().max()
                y_hat = upsampler(y_lowpass) / scaler
                band = y_hat.new_zeros((1, 513), dtype=torch.int64)
                band[:, :int(513 / q)] = 1
                shifted_gamma = gamma - 2 * scaler.log()
                norm_nlogsnr = (shifted_gamma + 20) / 40
                norm_nlogsnr.clip_(0, 1)
                y_recon = reverse_manifold(
                    y_hat, shifted_gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast(enabled=False)(lambda x, t, idx: model(
                        x, y_hat[:, idx], band, norm_nlogsnr[t:t+1])),
                    lr=lr,
                    verbose=False
                ) * scaler
            elif infer_type == "nuwave2-inpainting":
                scaler = y.abs().max()
                y_hat = upsampler(y_lowpass) / scaler
                band = y_hat.new_zeros((1, 513), dtype=torch.int64)
                band[:, :int(513 / q)] = 1
                shifted_gamma = gamma - 2 * scaler.log()
                norm_nlogsnr = (shifted_gamma + 20) / 40
                norm_nlogsnr.clip_(0, 1)
                y_recon = reverse(
                    y_hat, shifted_gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t: model(
                        x, y_hat, band, norm_nlogsnr[t:t+1])),
                    verbose=False
                ) * scaler
            elif infer_type == "nuwave2-ddim":
                scaler = y.abs().max()
                y_hat = upsampler(y_lowpass) / scaler
                band = y_hat.new_zeros((1, 513), dtype=torch.int64)
                band[:, :int(513 / q)] = 1
                y_recon = nuwave2_ddim(y_hat, band, gamma - 2 * scaler.log(),
                                       amp.autocast()(model),
                                       verbose=False) * scaler
            else:
                raise ValueError(
                    "infer_type must be one of 'nuwave', 'inpainting', 'nuwave-inpainting', 'nuwave-manifold', 'manifold', 'nuwave2', 'nuwave2-manifold', 'nuwave2-inpainting'")

            if offset:
                y_recon = torch.cat(
                    [y_recon, y_recon.new_zeros(1, offset)], dim=1)

            y_recon, raw_y = y_recon.squeeze(), raw_y.squeeze()

            assert not torch.isnan(y_recon).any(), "NaN detected"

            if target_sr is not None:
                y_recon = torch.from_numpy(
                    resample(y_recon.cpu().numpy(), target_sr / sr, 'sinc_best')).to(device)
                raw_y = torch.from_numpy(
                    resample(raw_y.cpu().numpy(), target_sr / sr, 'sinc_best')).to(device)

            lsd = evaluater(y_recon, raw_y).item()
            assert not math.isnan(lsd), "lsd is nan"
            rq.put((filename, lsd, y_recon, y_lowpass, sr))

    except Exception as e:
        rq.put((filename, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('cfg', type=str)
    parser.add_argument('vctk', type=str)
    parser.add_argument('--log-snr', type=str)
    parser.add_argument('--nuwave-ckpt', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--rate', type=int, default=2)
    parser.add_argument('-T', type=int, default=50)
    parser.add_argument('--infer-type', type=str,
                        choices=['inpainting', 'nuwave', 'nuwave-inpainting', 'manifold', 'nuwave-manifold', 'nuwave2', 'nuwave2-manifold', 'nuwave2-inpainting', 'nuwave2-ddim'], default='inpainting')
    parser.add_argument('--downsample-type', type=str,
                        choices=['sinc', 'stft', 'cheby'], default='stft')
    parser.add_argument('--lr', type=float, default=1.,
                        help="learning rate for manifold")
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--target-sr', type=int, default=None)

    args = parser.parse_args()

    set_start_method('spawn', force=True)

    gpus = torch.cuda.device_count()

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    cfg = OmegaConf.load(args.cfg)
    if 'nuwave2' in args.infer_type:
        model = module_arch.NuWave2(
            OmegaConf.create(
                {
                    'arch': {
                        'residual_layers': 15,
                        'residual_channels': 64,
                        'pos_emb_dim': 512,
                        'bsft_channels': 64
                    },
                    'dpm': {
                        'max_step': 1000,
                        'pos_emb_scale': 50000,
                        'pos_emb_channels': 128
                    },
                    'audio': {
                        'filter_length': 1024,
                        'hop_length': 256,
                        'win_length': 1024,
                        'sampling_rate': 48000
                    }
                }
            )
        )
        state_dict = torch.load(
            args.nuwave_ckpt, map_location=torch.device('cpu'))
        if not args.nuwave_ckpt.endswith('EMA'):
            state_dict = state_dict['state_dict']
        state_dict = dict((x[12:], y)
                          for x, y in state_dict.items() if x.startswith('model.model.'))
        model.load_state_dict(state_dict)
    elif 'nuwave' in args.infer_type:
        model = module_arch.NuWave()
        state_dict = torch.load(
            args.nuwave_ckpt, map_location=torch.device('cpu'))
        state_dict = dict((x[6:], y)
                          for x, y in state_dict.items() if x.startswith('model.'))
        model.load_state_dict(state_dict)
    else:
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(checkpoint['model' if args.raw else 'ema_model'])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    if cfg.train_T > 0:
        scheduler = module_arch.NoiseScheduler()
    else:
        scheduler = module_arch.LogSNRLinearScheduler()
    scheduler.load_state_dict(checkpoint['noise_scheduler'])
    scheduler.eval()
    scheduler = scheduler.cuda()

    if args.log_snr:
        gamma0, gamma1 = scheduler.gamma0.detach().cpu(
        ).numpy(), scheduler.gamma1.detach().cpu().numpy()
        log_snr = np.loadtxt(args.log_snr)
        xp = np.arange(len(log_snr))
        x = np.linspace(xp[0], xp[-1], args.T)
        gamma = -np.interp(x, xp, log_snr)
        steps = (gamma - gamma0) / (gamma1 - gamma0)
        gamma, steps = torch.tensor(gamma, dtype=torch.float32), torch.tensor(
            steps, dtype=torch.float32)
    else:
        t = torch.linspace(0, 1, args.T + 1).cuda()
        with torch.no_grad():
            gamma, steps = scheduler(t)

    file_q = Queue()
    result_q = Queue()
    processes = []

    for i in range(gpus):
        device = f'cuda:{i}'
        evaluater = LSD()
        if args.downsample_type == 'sinc':
            downsampler = Decimate(q=args.rate, **sinc_kwargs)
            upsampler = Upsample(q=args.rate, **sinc_kwargs)
        elif args.downsample_type == 'cheby':
            downsampler = ChebyDecimate(r=args.rate)
            upsampler = ChebyUpsample(r=args.rate)
        else:
            downsampler = STFTDecimate(args.rate)
            upsampler = Upsample(q=args.rate, **sinc_kwargs)

        p = Process(target=foo, args=(
            file_q, result_q, args.rate, args.infer_type, args.lr, args.target_sr,
            deepcopy(model).to(device), evaluater.to(device), downsampler.to(
                device), upsampler.to(device), gamma.to(device), steps.to(device)))
        processes.append(p)

    vctk_path = Path(args.vctk)
    test_files = list(vctk_path.glob('**/*.wav'))

    for filename in test_files:
        file_q.put(filename)

    for p in processes:
        p.start()

    pbar = tqdm(total=len(test_files))
    n = 0
    cumulated_lsd = 0
    try:
        while n < len(test_files):
            filename, lsd, *results = result_q.get()
            if isinstance(lsd, Exception):
                print(f'catch exception at {filename}: {lsd}')
                for p in processes:
                    p.terminate()
                break
            recon, lowpass, sr = results
            n += 1
            cumulated_lsd += (lsd - cumulated_lsd) / n
            pbar.set_description(
                f'LSD: {lsd:.4f}, Avg LSD: {cumulated_lsd:.4f}')
            pbar.update(1)

            if args.out_dir is not None:
                out_path = Path(args.out_dir) / filename.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(
                    out_path, recon.cpu().unsqueeze(0), sample_rate=args.target_sr if args.target_sr is not None else sr)

                out_path = Path(args.out_dir) / "inputs" / filename.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(out_path, lowpass.cpu(),
                                sample_rate=sr // args.rate)

    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        for p in processes:
            p.join()

    print(cumulated_lsd)
