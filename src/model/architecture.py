import torch
import torch.nn as nn
import src.config as cfg


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, use_rope=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = ApplyRoPE(self.head_dim, context_length)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, cache=None):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # reshape to multi head format
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # if cache exists, append new K, V to cached K, V
        if cache is not None:
            keys   = torch.cat([cache["key"],   keys],   dim=2)
            values = torch.cat([cache["value"], values], dim=2)
        if self.use_rope:
            queries = self.rope(queries)
            keys    = self.rope(keys)
        # update cache with latest full K, V
        new_cache = {"key": keys, "value": values}

        # total sequence length after appending cache
        total_seq_len = keys.shape[2]

        # attention scores: Q is only new tokens, K is full sequence
        attn_scores = queries @ keys.transpose(2, 3)

        # apply causal mask — only for training (no cache), during inference mask is not needed
        if cache is None:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, new_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
            use_rope=cfg.use_rope)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, new_cache = self.att(x, cache=cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, new_cache


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class ApplyRoPE(nn.Module):
    def __init__(self, head_dim, context_length):
        super().__init__()
        i = torch.arange(0, head_dim, 2).float()
        freqs = 1.0 / (10000 ** (i / head_dim))
        pos = torch.arange(context_length).float()
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [context_length, head_dim/2]
        self.register_buffer("angles", angles)
    def forward(self, x):
        # x shape: [batch, num_heads, seq_len, head_dim]
        seq_len = x.shape[2]

        # split into pairs
        x1 = x[..., 0::2]  # even dimensions [batch, heads, seq_len, head_dim/2]
        x2 = x[..., 1::2]  # odd dimensions  [batch, heads, seq_len, head_dim/2]

        # get cos and sin for current sequence length
        cos = torch.cos(self.angles[:seq_len, :])  # [seq_len, head_dim/2]
        sin = torch.sin(self.angles[:seq_len, :])  # [seq_len, head_dim/2]

        # apply rotation
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        # interleave x1_new and x2_new back together
        x_rotated = torch.stack([x1_new, x2_new], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [batch, heads, seq_len, head_dim]

        return x_rotated.type_as(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx, cache=None):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)

        if not cfg.use_rope:
            # learned positional embedding
            if cache is None or cache[0] is None:
                pos = torch.arange(seq_len, device=in_idx.device)
            else:
                past_len = cache[0]["key"].shape[2]
                pos = torch.arange(past_len, past_len + seq_len, device=in_idx.device)
            x = tok_embeds + self.pos_emb(pos)
        else:
            # rope handles position inside attention, no addition needed here
            x = tok_embeds

        x = self.drop_emb(x)

        # loop through blocks manually to pass cache per block
        new_cache = []
        for i, block in enumerate(self.trf_blocks):
            block_cache = cache[i] if cache is not None else None
            x, updated_cache = block(x, cache=block_cache)
            new_cache.append(updated_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, new_cache