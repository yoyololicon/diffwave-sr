import argparse
from pathlib import Path
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from kazane import Decimate, Upsample
from scipy.interpolate import interp1d
from omegaconf import OmegaConf
import models as module_arch
import numpy as np
from samplerate import resample

from vctk_infer import sinc_kwargs, LSD, STFTDecimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('cfg', type=str)
    parser.add_argument('vctk', type=str)
    parser.add_argument('--rate', type=int, default=2)

    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    cfg = OmegaConf.load(args.cfg)

    scheduler = module_arch.LogSNRLinearScheduler()
    scheduler.load_state_dict(checkpoint['noise_scheduler'])
    scheduler.eval()

    std = scheduler.gamma0.mul(0.5).exp().item()
    print(f'Noise std: {std}, gamma0: {scheduler.gamma0.item()}')

    evaluater = LSD()

    vctk_path = Path(args.vctk)
    test_files = list(vctk_path.glob('**/*.wav'))

    pbar = tqdm(test_files)
    n = 0
    cumulated_lsd = 0

    for filename in pbar:
        raw_y, sr = torchaudio.load(filename)
        raw_y = raw_y.squeeze()
        y_recon = raw_y + torch.randn_like(raw_y) * std

        # y_recon = torch.from_numpy(
        #     resample(y_recon.cpu().numpy(), 1 / args.rate, 'sinc_best'))
        # raw_y = torch.from_numpy(
        #     resample(raw_y.cpu().numpy(), 1 / args.rate, 'sinc_best'))

        lsd = evaluater(y_recon, raw_y, 1 / args.rate).item()

        n += 1
        cumulated_lsd += (lsd - cumulated_lsd) / n

        pbar.set_description(f'LSD: {lsd:.4f}, Avg LSD: {cumulated_lsd:.4f}')

    print(cumulated_lsd)
