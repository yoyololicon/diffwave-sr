import argparse
import os
import hydra
from omegaconf import OmegaConf
import torchaudio
import torch

from inference import reverse_process_ddim, reverse_process_new

import models as module_arch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('cfg', type=str)
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('-T', type=int, default=50)
    parser.add_argument('-N', type=int, default=3)
    parser.add_argument('--duration', type=float, default=10.0)

    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    cfg = OmegaConf.load(args.cfg)
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint['ema_model'])
    model.eval()
    model = model.cuda()
    sr = cfg.sr

    if cfg.train_T > 0:
        scheduler = module_arch.NoiseScheduler()
    else:
        scheduler = module_arch.LogSNRLinearScheduler()
    scheduler.load_state_dict(checkpoint['noise_scheduler'])
    scheduler.eval()
    scheduler = scheduler.cuda()

    t = torch.linspace(0, 1, args.T + 1).cuda()
    with torch.no_grad():
        gamma, steps = scheduler(t)

    z_T = torch.randn(args.N, int(sr * args.duration), device='cuda')

    with torch.no_grad():
        # x = reverse_process_new(z_T, gamma, steps, model, with_amp=True)
        x = reverse_process_ddim(z_T, gamma, steps, model, with_amp=True)

    for i, x_i in enumerate(x.cpu()):
        torchaudio.save(os.path.join(
            args.out_dir, f'{i}.wav'), x_i.unsqueeze(0), sr)
