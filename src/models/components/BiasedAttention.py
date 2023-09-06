import torch.nn as nn


class BiasedAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        q_bias=False,
        k_bias=False,
        v_bias=False,
        dropout=0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=q_bias)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=k_bias)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=v_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self, query, key, value, attn_booster=None, query_mask=None, key_mask=None
    ):
        N, B, C = query.shape

        q = (
            self.to_q(query * query_mask if query_mask is not None else query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.to_k(key * key_mask if key_mask is not None else key)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(value)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_booster is not None:
            # Repeat attn_mask for each head
            attn_booster = attn_booster.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn + attn_booster

        query = (attn @ v).transpose(1, 2).reshape(N, B, C)
        query = self.proj(query)
        query = self.proj_drop(query)

        # Average attn weights
        attn = attn.mean(dim=1)

        return query, attn
