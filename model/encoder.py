import torch.nn as nn
from torch.nn import Dropout
from .sublayers import MultiHeadAttention, PositionWiseFeedForwardNetwork


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):  # model_dim=512, num_heads=8, ff_dim=2018
        super(EncoderLayer, self).__init__()
        # Each sub-layer has an Add&Norm layer inside
        # 1) Self attention
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
        # 2) Positional wise feed forward network
        self.pos_ffn = PositionWiseFeedForwardNetwork(model_dim, ff_dim, dropout=dropout)

    def forward(self, enc_input, self_attn_mask):
        # 1) Self attention
        enc_output, self_attention = self.self_attn(
            enc_input, enc_input, enc_input,  # q, k, v: Encoder
            attn_mask=self_attn_mask)  # attn_mask = self_attn_mask = src_mask

        # 2) Feed forward network
        enc_output = self.pos_ffn(enc_output)

        return enc_output, self_attention


class Encoder(nn.Module):
    """A stack of 6 EncoderLayers."""

    def __init__(self, num_layers=6, model_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super(Encoder, self).__init__()
        # Encoder layers
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ff_dim, dropout=dropout) for _ in range(num_layers)])
        # Dropout
        self.dropout = Dropout(p=dropout)

    def forward(self, src_seq, src_mask, return_attentions=False):
        enc_output = src_seq

        enc_self_attentions = []
        for encoder_layer in self.layer_stack:
            enc_output, enc_self_attention = encoder_layer(enc_input=enc_output, self_attn_mask=src_mask)
            # Save attentions
            enc_self_attentions += enc_self_attention

        if return_attentions:  # If needed, please modify the transform's forward function.
            return enc_output, enc_self_attentions
        else:  # Default configuration
            return enc_output
