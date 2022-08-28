import torch
from torch import nn, Tensor
import torch.nn.functional as F


def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)


def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **config['args'], **kwargs)


def gamma2snr(g: Tensor) -> Tensor:
    return torch.exp(-g)


def snr2as(snr: Tensor):
    snr_p1 = snr + 1
    return torch.sqrt(snr / snr_p1), snr_p1.reciprocal()


def gamma2as(g: Tensor):
    var = g.sigmoid()
    return (1 - var).sqrt(), var


def gamma2logas(g: Tensor):
    log_var = -F.softplus(-g)
    return 0.5 * (-g + log_var), log_var


def beta2gamma(beta: Tensor):
    alpha = 1 - beta
    log_alpha = torch.log(alpha)
    log_alpha_hat = log_alpha.cumsum(dim=0)
    return torch.log(-torch.expm1(log_alpha_hat)) - log_alpha_hat
