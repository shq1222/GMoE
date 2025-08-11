import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum


class Residual(nn.Module):
    """Residual connection wrapper for any module."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Pre-normalization layer wrapper."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer module with multiple layers of attention and feed-forward networks."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Expert(nn.Module):
    """Expert module for processing spatial information."""
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.patch_size = 16
        self.num_frames = 16
        self.dim = 192
        self.depth = 6
        self.heads = 6
        self.pool = 'cls'
        self.dim_head = 64
        self.dropout = 0.
        self.emb_dropout = 0.
        self.scale_dim = 4

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_frames, (self.image_size // self.patch_size) ** 2 + 1, self.dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, 
            self.dim * self.scale_dim, self.dropout)
        self.Dropout = nn.Dropout(self.emb_dropout)

    def forward(self, x):
        b, t, n, _ = x.shape
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.Dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        return rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)


class Tower(nn.Module):
    """Tower module for processing temporal information and final classification."""
    def __init__(self, scale_dim=4):
        super().__init__()
        self.dim = 192
        self.depth = 6
        self.heads = 6
        self.pool = 'cls'
        self.dim_head = 64
        self.dropout = 0.
        self.emb_dropout = 0.
        self.scale_dim = 4
        self.num_classes = 4
        
        self.layer_norm = nn.LayerNorm(self.dim)
        self.linear = nn.Linear(self.dim, self.num_classes)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.temporal_transformer = Transformer(
            self.dim, self.depth, self.heads, self.dim_head,
            self.dim * scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)

    def forward(self, x):
        b, t, _ = x.shape
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.linear(self.layer_norm(x))


class GMoE(nn.Module):
    """Gated Mixture of Experts module for multi-task learning."""
    def __init__(self, num_experts, tasks, device):
        super().__init__()
        self.device = device
        self.image_size = 224
        self.patch_size = 16
        self.num_frames = 16
        self.dim = 192
        self.pool = 'cls'
        self.dim_head = 64
        self.dropout = 0.
        self.emb_dropout = 0.
        self.scale_dim = 4
        self.in_channels = 3
        
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = self.in_channels * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),
        )

        self.tasks = tasks
        self.dropout = nn.Dropout(self.emb_dropout)
        self.num_experts = num_experts
        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([Expert() for _ in range(self.num_experts)])
        self.towers = nn.ModuleList([Tower() for _ in range(self.tasks)])
        
        self.w_gates = nn.ParameterList([
            nn.Parameter(torch.randn(16 * 196 * self.dim, self.num_experts), requires_grad=True)
            for _ in range(self.tasks)
        ])

    def forward(self, x):
        x = self.to_patch_embedding(x)
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        b, t, n, k = x.to(self.device).shape
        gates_o = [self.softmax(x.reshape(b, -1) @ g.to(self.device)) for g in self.w_gates]
        
        # Combine expert outputs using gate weights
        tower_input = [
            o.t().unsqueeze(2).unsqueeze(3).expand(-1, -1, t, self.dim) * experts_o_tensor
            for o in gates_o
        ]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        
        # Process through task-specific towers
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output