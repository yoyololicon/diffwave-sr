import argparse
from pathlib import Path
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from kazane import Decimate, Upsample
from scipy.interpolate import interp1d

from vctk_infer import sinc_kwargs, LSD, STFTDecimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vctk', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--rate', type=int, default=2)
    parser.add_argument('--infer-type', type=str,
                        choices=['none', 'spline', 'linear'], default='none')
    parser.add_argument('--downsample-type', type=str,
                        choices=['sinc', 'stft'], default='stft')

    args = parser.parse_args()

    evaluater = LSD()
    if args.downsample_type == 'sinc':
        downsampler = Decimate(q=args.rate, **sinc_kwargs)
    else:
        downsampler = STFTDecimate(args.rate)

    if args.infer_type == 'none':
        upsampler = Upsample(q=args.rate, **sinc_kwargs)
    elif args.infer_type == 'linear':
        def upsampler(x):
            return F.interpolate(x.unsqueeze(1), scale_factor=args.rate, mode='linear', align_corners=False).squeeze(1)
    elif args.infer_type == 'spline':
        def upsampler(y):
            device = y.device
            dtype = y.dtype
            y = y.cpu().numpy()
            f = interp1d(np.arange(
                0, y.shape[-1] * args.rate, args.rate), y, axis=-1, kind='cubic', assume_sorted=True,
                bounds_error=False, fill_value='extrapolate')
            y_recon = f(np.arange(y.shape[-1] * args.rate))
            y_recon = torch.from_numpy(y_recon).to(device=device, dtype=dtype)
            return y_recon
    else:
        raise ValueError('Unknown infer type')

    vctk_path = Path(args.vctk)
    test_files = list(vctk_path.glob('**/*.wav'))

    pbar = tqdm(test_files)
    n = 0
    cumulated_lsd = 0

    for filename in pbar:
        raw_y, sr = torchaudio.load(filename)

        offset = raw_y.shape[1] % args.rate
        if offset:
            y = raw_y[:, :-offset]
        else:
            y = raw_y

        y_lowpass = downsampler(y)
        y_recon = upsampler(y_lowpass)

        if offset:
            y_recon = torch.cat(
                [y_recon, y_recon.new_zeros(1, offset)], dim=1)

        lsd = evaluater(y_recon.squeeze(), raw_y.squeeze()).item()

        n += 1
        cumulated_lsd += (lsd - cumulated_lsd) / n

        pbar.set_description(f'LSD: {lsd:.4f}, Avg LSD: {cumulated_lsd:.4f}')

        if args.out_dir is not None:
            out_path = Path(args.out_dir) / filename.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, y_recon.cpu(), sample_rate=sr)

            out_path = Path(args.out_dir) / "inputs" / filename.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, y_lowpass.cpu(),
                            sample_rate=sr // args.rate)

    print(cumulated_lsd)
