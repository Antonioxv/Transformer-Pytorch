import math

import torch
import torch.nn as nn
from torch.nn import Softmax, Linear, functional, Dropout, LayerNorm


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = None

    def forward(self, q, k, v, scale, attn_mask=None):
        """
        Calculate the Scaled dot-product attention, 在 encoder-decoder 的 Attention 层中 q_len(n) 和 k_len(m) 可能不同
        :param q: Query, shape: [batch_size, n_heads, q_len, q_dim]
        :param k: Key, shape: [batch_size, n_heads, k_len, k_dim]
        :param v: Value, shape: [batch_size, n_heads, v_len, v_dim]
        :param scale: Scale factor, a float number calculated by k_dim ^ 0.5
        :param attn_mask: Bool tensor, shape: [batch_size, n_heads, seq_len, seq_len]
        """

        # 1) MatMul
        # transpose the len and dim for k
        k_t = torch.transpose(k, -1, -2)
        attention = torch.matmul(q, k_t)  # shape: [32, 8, 10, 10]

        # 2) Scale
        attention = attention / scale

        # 3) Mask(opt.)
        if attn_mask is not None:  # if true, masked
            _INF = 1e9  # Softmax(-_INF) == 0
            attention = attention.masked_fill_(attn_mask == 0, -_INF)  # fill -_INF where mask == False

        # 4) SoftMax
        attention = functional.softmax(attention, dim=-1)

        # dropout
        if self.dropout is not None:
            attention = self.dropout(attention)

        # 5) MatMul
        context = torch.matmul(attention, v)  # shape: [32, 8, 10, 64]

        return context, attention


class MultiHeadAttention(nn.Module):
    """
    The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by
    a compatibility function of the query with the corresponding key.
    """

    def __init__(self, model_dim, num_heads, dropout=0.1):  # model_dim=512, num_heads=8
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = Dropout(p=dropout)
        self.norm = LayerNorm(model_dim)
        self.attn = ScaledDotProductAttention(self.dropout)

        # dim for each head
        self.q_dim = model_dim // num_heads  # q_dim = k_dim
        self.k_dim = model_dim // num_heads
        self.v_dim = model_dim // num_heads  # v_dim = k_dim

        # Linear layers
        self.w_q = Linear(model_dim, self.k_dim * num_heads)
        self.w_k = Linear(model_dim, self.k_dim * num_heads)
        self.w_v = Linear(model_dim, self.v_dim * num_heads)
        self.w_o = Linear(num_heads * self.v_dim, model_dim)

    def forward(self, q, k, v, attn_mask=None):
        """
        q_dim = k_dim = v_dim  = model_dim, k_len = v_len
        :param q: Query, shape: [batch_size, q_len, model_dim]
        :param k: Key, shape: [batch_size, k_len, model_dim]
        :param v: Value, shape: [batch_size, v_len, model_dim]
        :param attn_mask: Masking tensor, shape: [batch_size, seq_len, seq_len]
        :return:
        """

        # input
        residual = q

        # linear projections
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # split by heads
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.q_dim).transpose(1, 2)  # shape: [batch_size, num_heads, q_len, q_dim]
        k = k.view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)  # shape: [batch_size, num_heads, k_len, k_dim]
        v = v.view(batch_size, -1, self.num_heads, self.v_dim).transpose(1, 2)  # shape: [batch_size, num_heads, v_len, v_dim]

        # attn_mask -> 4dims -> repeat num_heads
        if attn_mask is not None:  # shape: [batch_size, seq_len, seq_len]
            # Head axis broadcasting, shape: [batch_size, num_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Scaled dot-product attention
        scale = math.sqrt(self.k_dim)
        context, attention = self.attn(q, k, v, scale, attn_mask)

        # concat heads
        context = context.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.v_dim * self.num_heads)  # shape: [batch_size, v_len, v_dim * num_heads]

        # final linear projection
        output = self.w_o(context)  # shape: [batch_size, v_len, model_dim]

        # add and norm layer
        output = self.norm(residual + output)

        # Output is computed as a weighted sum of values, has the same size as v.
        return output, attention  # shape: [32, 10, 512], [32, 10, 10]


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):  # model_dim=512, ff_dim=2048
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.dropout = Dropout(p=dropout)
        self.norm = LayerNorm(model_dim)

        # Single hidden layer
        self.w_1 = nn.Linear(model_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        """Simple implementation of ffn function."""
        # inputs
        residual = x

        # ffn function
        output = self.w_2(self.dropout(functional.relu(self.w_1(x))))

        # add and norm layer
        output = self.norm(residual + output)

        return output
