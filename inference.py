import torch
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm
from utils import gamma2snr, snr2as, gamma2as, gamma2logas


def reverse_process_new(z_1, gamma, steps, model, *condition, with_amp=False):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()

    T = gamma.numel()
    z_t = z_1
    for t in tqdm(range(T - 1, 0, -1)):
        s = t - 1
        with amp.autocast(enabled=with_amp):
            noise_hat = model(z_t, steps[t:t+1], *condition)
        noise_hat = noise_hat.float()
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]
        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    with amp.autocast(enabled=with_amp):
        noise_hat = model(z_t, steps[:1], *condition)
    noise_hat = noise_hat.float()
    final = (z_t - var[0].sqrt() * noise_hat) / log_alpha[0].exp()
    return final


def reverse_process_ddim(z_1, gamma, steps, model, with_amp=False):
    Pm1 = -torch.expm1((gamma[1:] - gamma[:-1]) * 0.5)
    log_alpha, log_var = gamma2logas(gamma)
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    std = log_var.mul(0.5).exp()

    T = gamma.numel() - 1
    z_t = z_1
    for t in tqdm(range(T, 0, -1)):
        s = t - 1
        with amp.autocast(enabled=with_amp):
            noise_hat = model(z_t, steps[t:t+1])
        noise_hat = noise_hat.float()
        z_t.mul_(alpha_st[s]).add_(std[s] * Pm1[s] * noise_hat)

    return z_t
