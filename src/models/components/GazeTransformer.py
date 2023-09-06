import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

from src.models.components.BiasedAttention import BiasedAttention


def _get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_copies)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class GazeTransformer(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        sa,
        ca,
        sa_mask,
        ca_mask,
        ca_pos,
        sa_pos,
        attn_booster,
    ):
        output = sa

        for layer in self.layers:
            output, sa_attn_weights, ca_attn_weights = layer(
                output,
                ca,
                sa_mask,
                ca_mask,
                ca_pos,
                sa_pos,
                attn_booster,
            )

        if self.norm is not None:
            output = self.norm(output)

        return (
            output.unsqueeze(0),
            sa_attn_weights.unsqueeze(0),
            ca_attn_weights.unsqueeze(0),
        )


class GazeTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = BiasedAttention(
            d_model,
            nhead,
            dropout=dropout,
        )
        self.multihead_attn = BiasedAttention(
            d_model,
            nhead,
            dropout=dropout,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        sa,
        ca,
        sa_mask,
        ca_mask,
        ca_pos,
        sa_pos,
        attn_booster,
    ):
        q = k = self.with_pos_embed(sa, sa_pos)
        tgt2, sa_attn_weights = self.self_attn(
            query=q,
            key=k,
            value=sa,
            query_mask=sa_mask,
            key_mask=sa_mask,
            attn_booster=attn_booster,
        )
        sa = sa + self.dropout1(tgt2)
        sa = self.norm1(sa)

        tgt2, ca_attn_weights = self.multihead_attn(
            query=self.with_pos_embed(sa, sa_pos),
            key=self.with_pos_embed(ca, ca_pos),
            value=ca,
            query_mask=sa_mask,
            key_mask=ca_mask,
            attn_booster=attn_booster,
        )

        sa = sa + self.dropout2(tgt2)
        sa = self.norm2(sa)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(sa))))
        sa = sa + self.dropout3(tgt2)
        sa = self.norm3(sa)

        return sa, sa_attn_weights, ca_attn_weights
