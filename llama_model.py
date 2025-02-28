# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)
        self.wk = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)
        self.wv = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)
        self.wo = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)


    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).contiguous()
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).contiguous()
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).contiguous()

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)


        xq = xq.transpose(1, 2).contiguous()  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2).contiguous()  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xv = xv.transpose(1, 2).contiguous()  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3).contiguous()) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1).contiguous()
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLamaTransformer(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 vocab_size: int = -1,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 norm_eps: float = 1e-5,
                 rope_theta: float = 500000,
                 max_batch_size: int = 32,
                 max_seq_len: int = 128,
                 ):
        super().__init__()
        self.args = ModelArgs(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.transformer_layers = torch.nn.ModuleList(
            [TransformerBlock(layer_id, self.args) for layer_id in range(self.args.n_layers)]
        )
        self.rms_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len,
            self.args.rope_theta,
        ).to("cuda")

    def forward(self, embed: torch.Tensor):
        # self.freqs_cis = self.freqs_cis.to(embed.device)
        for layer in self.transformer_layers:
            embed = layer(embed, 0, self.freqs_cis, mask=None)
        embed = self.rms_norm(embed)
        return embed


class LLamaTransformerV2(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 vocab_size: int = -1,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 norm_eps: float = 1e-5,
                 rope_theta: float = 500000,
                 max_batch_size: int = 32,
                 max_seq_len: int = 128,
                 ):
        super().__init__()
        self.args = ModelArgs(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.transformer_layers = torch.nn.ModuleList(
            [TransformerBlock(layer_id, self.args) for layer_id in range(self.args.n_layers)]
        )
        self.rms_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.register_buffer('freqs_cis', precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len,
            self.args.rope_theta,
        ))

    def forward(self, embed: torch.Tensor):
        for layer in self.transformer_layers:
            embed = layer(embed, 0, self.freqs_cis, mask=None)
        embed = self.rms_norm(embed)
        return embed

