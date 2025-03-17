import einops
import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x, *args):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply standard scaled dot-product attention."""
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(-2, -1)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


class LinearCrossAttention(nn.Module):
    def __init__(self, input_dim: int, conditioning_dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """
        Linearized Cross Attention: input attends to conditioning

        Args:
            input_dim (int): Channels in input
            conditioning_dim (int): Channels in conditioning
            heads (int, optional): Attention Heads. Defaults to 4.
            dim_head (int, optional): Channels per head. Defaults to 32.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv1d(input_dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(conditioning_dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # E: embeddings pace
        # S: sequence length
        # H: number of heads

        # [B, H*E, S_i]
        q = self.to_q(x)
        # [B, H*E, S_c]
        k, v = self.to_kv(c).chunk(2, dim=1)

        # [B, H, E, S]
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), (q, k, v))
        q = q * self.scale

        # [B, H, E, E]
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # [B, H, E, S_i] -> [B, H*E, S_i]
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, input_dim: int, conditioning_dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """
        Linearized Cross Attention: input attends to conditioning

        Args:
            input_dim (int): Channels in input
            conditioning_dim (int): Channels in conditioning
            heads (int, optional): Attention Heads. Defaults to 4.
            dim_head (int, optional): Channels per head. Defaults to 32.
        """
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv1d(input_dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(conditioning_dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # E: embeddings pace
        # S: sequence length
        # H: number of heads

        # [B, H*E, S_i]
        q = self.to_q(x)
        # [B, H*E, S_c]
        k, v = self.to_kv(c).chunk(2, dim=1)

        # [B, H, E, S]
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), (q, k, v))

        # Compute attention weighting between q and k, [B, H, S_i, S_c]
        attention = torch.einsum('b h d n, b h d m -> b h n m', q, k)
        attention *= self.scale
        attention = attention.softmax(dim=-1)

        # [B, H, E, S_i] -> [B, H*E, S_i]
        out = torch.einsum('b h n m, b h d m -> b h d n', attention, v)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')

        return self.to_out(out)
