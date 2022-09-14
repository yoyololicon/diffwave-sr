from math import inf
from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .model import DiffusionEmbedding


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = pos_seq[:, None] * self.inv_freq
        pos_emb = torch.view_as_real(
            torch.exp(1j * sinusoid_inp)).view(pos_seq.size(0), -1)
        return pos_emb


class RelMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias=False, add_bias_kv=False, add_zero_attn=False,
                         kdim=None, vdim=None, batch_first=True, **kwargs)
        self.register_parameter("u", nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim)))
        self.register_parameter("v", nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim)))
        self.pos_emb = PositionalEmbedding(self.embed_dim)
        self.pos_emb_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, C]
        B, T, _ = x.size()
        seq = torch.arange(-T + 1, T, device=x.device)
        pos_emb = self.pos_emb_proj(self.pos_emb(seq))
        pos_emb = pos_emb.view(-1, self.num_heads, self.head_dim)

        h = x @ self.in_proj_weight.t()
        h = h.view(B, T, self.num_heads, self.head_dim * 3)
        w_head_q, w_head_k, w_head_v = h.chunk(3, dim=-1)

        rw_head_q = w_head_q + self.u
        AC = rw_head_q.transpose(1, 2) @ w_head_k.permute(0, 2, 3, 1)

        rr_head_q = w_head_q + self.v
        BD = rr_head_q.transpose(1, 2) @ pos_emb.permute(1,
                                                         2, 0)       # [B, H, T, 2T-1]
        BD = F.pad(BD, (1, 1)).view(B, self.num_heads, 2 *
                                    T + 1, T)[:, :, 1::2, :]  # [B, H, T, T]

        attn_score = (AC + BD) / self.head_dim ** 0.5

        if mask is not None:
            attn_score = attn_score.masked_fill(mask, -inf)

        with torch.cuda.amp.autocast(enabled=False):
            attn_prob = F.softmax(attn_score.float(), dim=-1)
        attn_prob = F.dropout(attn_prob, self.dropout, self.training)

        attn_vec = attn_prob @ w_head_v.transpose(1, 2)
        return self.out_proj(attn_vec.permute(0, 2, 1, 3).reshape(B, T, -1))


class RelEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,  d_model: int, nhead: int, *args, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, nhead, *args, batch_first=True, dropout=dropout, **kwargs)
        self.self_attn = RelMultiheadAttention(d_model, nhead, dropout=dropout)

    def _sa_block(self, x: Tensor, mask: Tensor = None) -> Tensor:
        return self.dropout1(self.self_attn(x, mask))

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


@torch.jit.script
def glu(a, b):
    return a * b.sigmoid()


@torch.jit.script
def standardize(x, mu, std):
    return (x - mu) / std


@torch.jit.script
def destandardize(x, mu, std):
    return x * std + mu


def rescale_conv(reference):
    @torch.no_grad()
    def closure(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            std = m.weight.std()
            scale = (std / reference) ** 0.5
            m.weight.div_(scale)
            if m.bias is not None:
                m.bias.div_(scale)
    return closure


class Demucs(nn.Module):
    def __init__(self,
                 channels=64,
                 depth=6,
                 rescale=0.1,
                 kernel_size=8,
                 stride=4,
                 attention_layers=4):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth
        self.channels = channels

        self.chunk_size = stride ** depth

        self.pre_encoder = nn.ModuleList()
        self.pre_decoder = nn.ModuleList()
        self.post_decoder = nn.ModuleList()

        self.diffusion_embedding = DiffusionEmbedding(1)
        self.enc_diff_proj = nn.ModuleList()
        self.dec_diff_proj = nn.ModuleList()

        in_channels = 1
        for index in range(depth):
            self.pre_encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, channels, kernel_size, stride, padding=kernel_size // 4
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels * 2, 1)
                )
            )
            self.enc_diff_proj.append(nn.Linear(512, channels * 2, bias=False))

            decode = []
            out_channels = in_channels
            self.pre_decoder.insert(
                0,
                nn.Conv1d(channels, channels * 2, 3, padding=1, bias=False)
            )
            self.dec_diff_proj.insert(
                0, nn.Linear(512, channels * 2, bias=False))
            decode = [
                nn.ConvTranspose1d(channels, out_channels,
                                   kernel_size, stride, padding=kernel_size // 4)
            ]
            if index > 0:
                decode.append(nn.ReLU(inplace=True))
            self.post_decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels *= 2

        channels = in_channels

        encoder_layer = RelEncoderLayer(
            d_model=channels, nhead=16, dim_feedforward=channels * 4,
            dropout=0
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, attention_layers)

        self.apply(rescale_conv(reference=rescale))

    def forward(self, x, diffusion_step, context: int = None):
        diffusion_embs = self.diffusion_embedding(diffusion_step)
        x = x * x.pow(2).mean(dim=-1, keepdim=True).rsqrt()
        x = x.unsqueeze(1)

        offset = x.size(2) % self.chunk_size
        if offset > 0:
            x = F.pad(x, (0, self.chunk_size - offset), 'reflect')

        saved = []
        for encode, proj in zip(self.pre_encoder, self.enc_diff_proj):
            x = encode(x) + proj(diffusion_embs).unsqueeze(-1)
            x = F.glu(x, 1)
            saved.append(x)

        x = x.transpose(1, 2)
        mask = None
        if context and context < x.size(1):
            mask = x.new_ones(x.size(1), x.size(1), dtype=torch.bool)
            mask = torch.triu(mask, diagonal=context)
            mask = mask | mask.T

        x = self.encoder(x, mask).transpose(1, 2)

        for pre, post, proj in zip(self.pre_decoder, self.post_decoder, self.dec_diff_proj):
            skip = saved.pop()
            x = x + skip
            x = pre(x) + proj(diffusion_embs).unsqueeze(-1)
            x = F.glu(x, 1)
            x = post(x)

        if offset > 0:
            x = x[..., :offset - self.chunk_size]

        return x.squeeze(1)
