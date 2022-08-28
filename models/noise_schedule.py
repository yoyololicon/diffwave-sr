import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize


class Nonnegative(nn.Module):
    def forward(self, X):
        return X.abs()


class NoiseScheduler(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = parametrize.register_parametrization(
            nn.Linear(1, 1, bias=True), 'weight', Nonnegative())
        self.l2 = parametrize.register_parametrization(
            nn.Linear(1, 1024, bias=True), 'weight', Nonnegative())
        self.l3 = parametrize.register_parametrization(
            nn.Linear(1024, 1, bias=False), 'weight', Nonnegative())

        self.gamma1 = nn.Parameter(torch.ones(1) * 0, requires_grad=True)
        self.gamma0 = nn.Parameter(torch.ones(1) * -10, requires_grad=True)
        self.register_buffer('t01', torch.tensor([0., 1.]))

    def gamma_hat(self, t: torch.Tensor):
        l1 = self.l1(t)
        return l1 + self.l3(self.l2(l1).sigmoid())

    def forward(self, t: torch.Tensor):
        t = t.clamp(0, 1)
        min_gamma_hat, max_gamma_hat,  gamma_hat = self.gamma_hat(
            torch.cat([self.t01, t], dim=0).unsqueeze(-1)).squeeze(1).split([1, 1, t.numel()], dim=0)
        gamma0, gamma1 = self.gamma0, self.gamma1
        normalized_gamma_hat = (gamma_hat - min_gamma_hat) / \
            (max_gamma_hat - min_gamma_hat)
        # gamma = gamma_hat
        gamma = gamma0 + (gamma1 - gamma0) * normalized_gamma_hat

        return gamma, normalized_gamma_hat


class LogSNRLinearScheduler(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma1 = nn.Parameter(torch.ones(1) * 0, requires_grad=True)
        self.gamma0 = nn.Parameter(torch.ones(1) * -10, requires_grad=True)

    def forward(self, t: torch.Tensor):
        t = t.clamp(0, 1)
        gamma = self.gamma0 * (1 - t) + self.gamma1 * t
        return gamma, t


class CosineScheduler(nn.Module):
    def __init__(self, gamma0: float = -23, gamma1: float = 3.6):
        super().__init__()

        self.register_buffer('gamma0', torch.tensor(gamma0))
        self.register_buffer('gamma1', torch.tensor(gamma1))

    def schedule(self, t: torch.Tensor):
        return torch.tan(torch.pi * t * 0.5).reciprocal_().pow(2)

    def gamma2t(self, gamma: torch.Tensor):
        return 2 * torch.atan(gamma.mul(0.5).exp()) / torch.pi

    def forward(self, t: torch.Tensor):
        min_t = self.gamma2t(self.gamma0)
        max_t = self.gamma2t(self.gamma1)
        t = min_t + (max_t - min_t) * t.clamp(0, 1)

        snr = self.schedule(t)
        r = -snr.log()
        return r, torch.clip_((r - self.gamma0) / (self.gamma1 - self.gamma0), 0, 1)


if __name__ == '__main__':
    from torch.autograd import grad
    gamma = NoiseScheduler()

    t = torch.arange(10) / 9
    t = torch.tensor(t,  requires_grad=True)
    g = gamma(t)
    print(t, g)

    print(grad(g.sum(), t, only_inputs=True))
    # g.sum().backward()
