import argparse
import json
import torch
import torchaudio
from functools import partial

import models as module_arch
from utils.utils import get_instance
from inference import *


def main(config, ckpt, infile, outfile, T, amp, deterministic):
    device = torch.device('cuda')
    trainer_config = config['trainer']
    ckpt_dict = torch.load(ckpt, map_location=device)
    n_fft = trainer_config['n_fft']
    hop_length = trainer_config['hop_length']
    n_mels = trainer_config['n_mels']
    sr = trainer_config['sr']
    train_T = trainer_config['train_T']
    model = get_instance(module_arch, config['arch']).to(device)
    mel_spec = module_arch.MelSpec(sr, n_fft, hop_length=hop_length,
                                   f_min=20, f_max=8000, n_mels=n_mels).to(device)
    model.load_state_dict(ckpt_dict['ema_model'])

    if 'noise_scheduler' in ckpt_dict:
        noise_scheduler = module_arch.NoiseScheduler().to(device)
        noise_scheduler.load_state_dict(
            ckpt_dict['noise_scheduler'], strict=False)
        noise_scheduler.eval()
    else:
        max_log_snr = trainer_config['max_log_snr']
        min_log_snr = trainer_config['min_log_snr']
        noise_scheduler = module_arch.CosineScheduler(
            gamma0=-max_log_snr, gamma1=-min_log_snr).to(device)
    model.eval()

    y, sr = torchaudio.load(infile)
    y = y.mean(0, keepdim=True).to(device)
    mels = mel_spec(y)

    z_1 = torch.randn_like(y)

    if train_T:
        steps = torch.linspace(0, train_T, T + 1,
                               device=device).round().long()
        gamma, steps = noise_scheduler(steps / train_T)
    else:
        steps = torch.linspace(0, 1, T + 1, device=device)
        gamma, steps = noise_scheduler(steps)

    infer_func = partial(model, spectrogram=mels)

    with torch.no_grad():
        if deterministic:
            z_0 = reverse_process_ddim(z_1, gamma, steps, infer_func, with_amp=amp)
        else:
            z_0 = reverse_process_new(z_1, gamma, steps, infer_func, with_amp=amp)

    x = z_0.squeeze().clip(-0.99, 0.99)
    torchaudio.save(outfile, x.unsqueeze(0).cpu(), sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferencer')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('ckpt', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-T', type=int, default=20)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ddim', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.ckpt, args.infile, args.outfile, args.T, args.amp, args.ddim)
