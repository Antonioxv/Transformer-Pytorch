'''
Date: 2022-02-11 14:33:56
LastEditors: fuchaoxin
LastEditTime: 2022-02-11 09:54:54
FilePath: \Visual Studiod:\Desktop\idsm\part2\transformer\transformer-pytorch\utils\utils.py
'''
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm, Dropout
from torch.autograd import Variable


class VocabEmbedding(nn.Module):
    def __init__(self, num_vocab, vocab_emb_dim, padding_idx):
        super(VocabEmbedding, self).__init__()
        self.num_embeddings = num_vocab
        self.embedding_dim = vocab_emb_dim
        self.lut = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=padding_idx)  # default: padding_idx=0

    def forward(self, x):
        """In the embedding layers, we multiply those weights by vocab_emb_dim^0.5."""
        return self.lut(x) * (self.embedding_dim ** 0.5)  # shape: [num_vocab, vocab_emb_dim]


class PositionalEncoding(nn.Module):
    """Or positional embedding, a simple implementation of PE function, is not trainable."""

    def __init__(self, pos_emb_dim, max_seq_len, dropout):  # pos_emb_dim = mode_dim
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        # pe function
        pe = torch.zeros(max_seq_len, pos_emb_dim)  # shape: [max_seq_len, pos_emb_dim]
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # shape: [max_seq_len] -> [max_seq_len, 1]
        divided = torch.exp(-1 * (torch.arange(0, pos_emb_dim, 2) / pos_emb_dim) * math.log(10000.0))
        pe[:, 0::2] = torch.sin(pos * divided)  # 2i
        pe[:, 1::2] = torch.cos(pos * divided)  # 2i + 1
        pe = pe.unsqueeze(0)  # shape: [1, max_seq_len, pos_emb_dim]

        self.register_buffer('pe', pe)  # not trainable

    def forward(self, x):
        """Just add."""
        output = x + self.pe[:, :x.size(1)]
        return self.dropout(output)


class SequenceEmbedding(nn.Module):
    def __init__(self, num_vocab, vocab_emb_dim, padding_idx, pos_emb_dim, max_seq_len, dropout):
        super(SequenceEmbedding, self).__init__()
        self.vocab_emb = VocabEmbedding(num_vocab=num_vocab, vocab_emb_dim=vocab_emb_dim, padding_idx=padding_idx)
        self.pos_emb = PositionalEncoding(pos_emb_dim=pos_emb_dim, max_seq_len=max_seq_len, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Embedding for input seq
        :param x: shape: [32, 10]
        :return: shape: [32, 10, 512]
        """
        output = self.vocab_emb(x)  # shape: [32, 10, 512]
        output = self.pos_emb(output)
        return self.dropout(output)


def padding_mask(seq, padding_idx):
    """
    Create a mask to hide padding.
    :param seq: shape: [batch_size, seq_len]
    :param padding_idx: vocab embedding padding index
    :return: mask: shape: [[batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.size()
    mask = (seq != padding_idx).unsqueeze(-2)
    return mask.expand(batch_size, seq_len, seq_len)  # shape: [32, 10, 10]


def subsequent_mask(seq):
    """
    Mask out subsequent positions.
    Below the attention mask shows the position each tgt word (row) is allowed to look at (column).
    Words are blocked for attending to future words during training.
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
    """
    batch_size, seq_len = seq.size()
    attention_shape = (batch_size, seq_len, seq_len)
    mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')  # shape: [32, 10, 10]
    return torch.from_numpy(mask) == 0  # 0 -> True, others -> False


class LabelSmoothing(nn.Module):
    def __init__(self, epsilon=0.1):
        """
        Applies label smoothing. See https://arxiv.org/abs/1512.00567.
        :param epsilon: Smoothing rate.
        """
        super(LabelSmoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1 - self.epsilon) * inputs) + (self.epsilon / K)
