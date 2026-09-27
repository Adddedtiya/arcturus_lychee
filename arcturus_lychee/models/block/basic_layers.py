"""Transformer blocks: feed-forward, attention, and a full transformer.

The input shape of all blocks is [batch, tokens, dim]. The output has the
same shape. These blocks are a start for projects that need attention.
"""

import torch
import torch.nn as nn

from einops import rearrange


class BasicFeedForward(nn.Module):
    """LayerNorm, then Linear -> GELU -> Linear, with dropout."""

    def __init__(self, dim : int, hidden_dim : int, dropout : float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicAttention(nn.Module):
    """Multi-head self-attention with an explicit softmax.

    This version calculates the attention matrix itself. Thus it is easy to
    read, but it uses more memory than BasicAttentionWithSDPA.
    """

    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout : float = 0.0) -> None:
        super().__init__()
        inner_dim   = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv     = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BasicAttentionWithSDPA(nn.Module):
    """Multi-head self-attention with torch scaled_dot_product_attention (SDPA).

    SDPA selects a fast kernel, for example FlashAttention, if one is available.
    With dropout 0 and the same weights, the result is the same as BasicAttention.
    The memory use is less.
    """

    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout : float = 0.0) -> None:
        super().__init__()
        inner_dim   = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads         = heads
        self.dropout_value = dropout

        self.norm   = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv     = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n l (h e) -> n h l e', h = self.heads), qkv)

        # SDPA applies the scale 1 / sqrt(head_dim) itself. The attention has no mask.
        out = nn.functional.scaled_dot_product_attention(
            query     = q,
            key       = k,
            value     = v,
            dropout_p = self.dropout_value if self.training else 0.0,
            is_causal = False,    # Set True for an autoregressive model.
        )

        out = rearrange(out, 'n h l e -> n l (h e)')
        return self.to_out(out)


class BasicTransformer(nn.Module):
    """A stack of depth blocks. Each block has attention and a feed-forward part, with residual connections."""

    def __init__(
            self,
            dim      : int,
            depth    : int,
            heads    : int,
            dim_head : int,
            mlp_dim  : int,
            dropout  : float = 0.0,
        ) -> None:
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BasicAttentionWithSDPA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                BasicFeedForward(dim, mlp_dim, dropout = dropout),
            ]))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)   + x
        return self.norm(x)


if __name__ == "__main__":
    # Demo: one forward pass on the GPU, or on the CPU if no GPU is available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicTransformer(
        dim      = 8,
        depth    = 2,
        heads    = 8,
        dim_head = 16,
        mlp_dim  = 128,
    ).to(device)

    x = torch.rand(1, 1024, 8, device = device, requires_grad = True)
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
