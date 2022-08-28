import torch
import torch.nn.functional as F
import math

from utils import gamma2as, gamma2logas


def diffusion_elbo(gamma_0, gamma_1, d_gamma_t,
                   x, noise, noise_hat):
    log_alpha_0, log_var_0 = gamma2logas(gamma_0)
    log_alpha_1, log_var_1 = gamma2logas(gamma_1)

    # prior loss KL(q(z_1|x) || p(z_1)))
    x_flat = x.view(-1)
    x_dot = x_flat @ x_flat / x_flat.numel()
    prior_loss = 0.5 * (log_var_1.exp() + x_dot *
                        torch.exp(log_alpha_1 * 2) - 1 - log_var_1)

    # recon loss E[-log p(x | z_0)]
    ll = -0.5 * (gamma_0 + 1 + math.log(2 * math.pi))
    recon_loss = -ll

    extra_dict = {
        'kld': prior_loss.item(),
        'll': ll.item()
    }
    # diffusion loss
    diff = noise - noise_hat
    loss_T_raw = 0.5 * (d_gamma_t * (diff * diff).mean(1)
                        ) / x.shape[0]
    loss_T = loss_T_raw.sum()
    extra_dict['loss_T_raw'] = loss_T_raw.detach()
    extra_dict['loss_T'] = loss_T.item()

    loss = prior_loss + recon_loss + loss_T
    elbo = -loss
    extra_dict['elbo'] = elbo.item()
    return loss, extra_dict
