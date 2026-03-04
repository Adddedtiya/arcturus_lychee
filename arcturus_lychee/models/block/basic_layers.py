import torch
import torch.nn as nn

from einops     import rearrange, reduce, repeat

class BasicFeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BasicAttention(nn.Module):
    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class BasicTransformer(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, dim_head : int, mlp_dim : int, dropout = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BasicAttentionWithSDPA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                BasicFeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)   + x
        return self.norm(x)


class BasicAttentionWithSDPA(nn.Module):
    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.norm = nn.LayerNorm(dim)

        self.attend  = nn.Softmax(dim = -1)
        self.dropout_value = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n l (h e) -> n h l e', h = self.heads), qkv)

        # 2. Use SDPA with the bias passed as the mask
        # Note: scale is handled automatically by SDPA (1/sqrt(head_dim))
        out = nn.functional.scaled_dot_product_attention(
            query       = q,
            key         = k,
            value       = v, 
            dropout_p   = self.dropout_value if self.training else 0.0,
            is_causal   = False  # Set True if using for an autoregressive model
        )

        out = rearrange(out, 'n h l e -> n l (h e)')
        out = self.to_out(out)
        return out



if __name__ == "__main__":
    print("Basic Layers Modules !")

    model = BasicTransformer(
        dim   = 8,
        depth = 2,
        heads = 8,
        dim_head = 16,
        mlp_dim  = 128,
    ).to('cuda')
    
    x = torch.rand(1, 1024, 8, requires_grad = True).to('cuda')
    y : torch.Tensor = model(x)
    print(y.shape)