import torch.nn as nn
from torch.nn import Dropout
from .sublayers import MultiHeadAttention, PositionWiseFeedForwardNetwork


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):  # model_dim=512, num_heads=8, ff_dim=2048
        super(DecoderLayer, self).__init__()
        # Each sub-layer has an Add&Norm layer inside
        # 1) Self attention
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
        # 2) Encoder decoder attention
        self.enc_dec_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
        # 3) Positional wise feed forward network
        self.pos_ffn = PositionWiseFeedForwardNetwork(model_dim, ff_dim, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask, enc_dec_attn_mask):
        # 1) Self attention
        dec_output, dec_self_attention = self.self_attn(
            dec_input, dec_input, dec_input,  # q, k, v: Decoder
            attn_mask=self_attn_mask)  # attn_mask = self_attn_mask = tgt_mask

        # 2) Encoder decoder attention
        dec_output, enc_dec_attention = self.enc_dec_attn(
            dec_output, enc_output, enc_output,  # k, v: Encoder, q: Decoder
            attn_mask=enc_dec_attn_mask)  # attn_mask = enc_dec_attn_mask = src_mask

        # 3) Positional wise feed forward network
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_self_attention, enc_dec_attention


class Decoder(nn.Module):
    """A stack of 6 DecoderLayers."""

    def __init__(self, num_layers=6, model_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super(Decoder, self).__init__()
        # Decoder layers
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        # Dropout
        self.dropout = Dropout(p=dropout)

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask, return_attentions=False):
        dec_output = tgt_seq

        dec_self_attentions, enc_dec_attentions = [], []
        for dec_layer in self.layer_stack:
            dec_output, self_attention, enc_dec_attention = dec_layer(
                dec_input=dec_output, enc_output=enc_output,
                self_attn_mask=tgt_mask, enc_dec_attn_mask=src_mask)
            # Save attentions
            dec_self_attentions += self_attention
            enc_dec_attentions += enc_dec_attention

        if return_attentions:  # If needed, please modify the transform's forward function.
            return dec_output, dec_self_attentions, enc_dec_attentions
        else:  # Default configuration
            return dec_output
