import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head causal self-attention with separate q/k/v/out projections
    (kept separate, rather than one fused qkv matrix, so LoRA can target
    individual projections by name)."""

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Small character-level decoder-only transformer, just big enough to
    make the pretrain -> adapt experiment meaningful without needing a
    downloaded pretrained checkpoint."""

    def __init__(self, vocab_size, dim=128, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
