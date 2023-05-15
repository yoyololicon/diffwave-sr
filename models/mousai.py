import torch
from torch import nn
import torch.nn.functional as F
from audio_diffusion_pytorch import UNetV0

from .model import DiffusionEmbedding


class Mousai(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.unet = UNetV0(dim=1, in_channels=1, **kwargs)

    def forward(self, audio, diffusion_step, _=None):
        return self.unet(audio.unsqueeze(1), diffusion_step).squeeze(1)
