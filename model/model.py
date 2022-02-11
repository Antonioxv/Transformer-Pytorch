import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from torch.nn import Dropout, functional
from utils import SequenceEmbedding, padding_mask, subsequent_mask


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, model_dim, num_vocab):
        super(Generator, self).__init__()
        self.num_vocab = num_vocab
        self.proj = nn.Linear(model_dim, num_vocab)

    def forward(self, x):  # shape: [32, 10, 512]
        logits = self.proj(x)  # shape: [32, 10, num_tgt_vocab]
        probs = functional.softmax(logits, dim=-1).view(-1, self.num_vocab)  # shape: [320, num_tgt_vocab]
        return logits, probs


class Transformer(nn.Module):
    """Implementation of the transformer in pytorch, author: Chaoxin FU."""

    def __init__(self, num_src_vocab, num_tgt_vocab,
                 src_padding_idx=0, tgt_padding_idx=0, vocab_emb_dim=512, max_seq_len=10,
                 model_dim=512, k_dim=64, v_dim=64, ff_dim=2048, num_heads=8, num_layers=6, dropout=0.1):
        """
        Initialize the transformer.
        :param num_src_vocab: Number of the source vocab(language: DE)
        :param num_tgt_vocab: Number of the target vocab(language: EN)
        :param src_padding_idx: Padding index for mask, usually 0
        :param tgt_padding_idx: Padding index for mask, usually 0
        :param vocab_emb_dim: Vocab embedding dim
        :param max_seq_len: Maximum number of words in a sentence
        :param model_dim: 512
        :param k_dim: 512/8
        :param v_dim: 512/8
        :param ff_dim: 2048
        :param num_heads: 8
        :param num_layers: Number of encoder/decoder layers
        :param dropout: 0.1, borrowed from harvardnlp's implementation
        """
        # q_dim = k_dim = v_dim = model_dim / num_heads
        super(Transformer, self).__init__()
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

        # 1) Embedding
        self.src_emb = SequenceEmbedding(num_vocab=num_src_vocab + 1, vocab_emb_dim=vocab_emb_dim, padding_idx=src_padding_idx,
                                         pos_emb_dim=model_dim, max_seq_len=max_seq_len, dropout=dropout)
        self.tgt_emb = SequenceEmbedding(num_vocab=num_tgt_vocab + 1, vocab_emb_dim=vocab_emb_dim, padding_idx=tgt_padding_idx,
                                         pos_emb_dim=model_dim, max_seq_len=max_seq_len, dropout=dropout)

        # 2) Encoder, decoder and generator
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
                               ff_dim=ff_dim, dropout=dropout)
        self.decoder = Decoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
                               ff_dim=ff_dim, dropout=dropout)
        self.generator = Generator(model_dim=model_dim, num_vocab=num_tgt_vocab)

        # 3) Dropout
        self.dropout = Dropout(p=dropout)

    def forward(self, src_seq, tgt_seq):
        """
        Take in src and tgt sequences and process embeddings and masks.
        :param src_seq: source sequence, shape: [batch_size, src_len]
        :param tgt_seq: target sequence, shape: [batch_size, tgt_len]
        :return: Output of the transformer model.
        """
        # 1) Embedding
        # 1.1) Encoder
        enc_input = self.src_emb(src_seq)
        # 1.2) Decoder
        dec_input = self.tgt_emb(tgt_seq)

        # 2) Mask
        # 2.1) Encoder
        src_mask = padding_mask(src_seq, self.src_padding_idx)
        # 2.2) Decoder
        tgt_mask = padding_mask(tgt_seq, self.tgt_padding_idx) & subsequent_mask(tgt_seq)

        logits, probs = self.generator(self.decode(dec_input, tgt_mask, self.encode(enc_input, src_mask), src_mask))

        return logits, probs

    def encode(self, src_seq, src_mask):
        return self.encoder(src_seq, src_mask)

    def decode(self, enc_output, src_mask, tgt_seq, tgt_mask):
        return self.decoder(tgt_seq, tgt_mask, enc_output, src_mask)
