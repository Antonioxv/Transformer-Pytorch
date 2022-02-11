"""
Code derived and rehashed from: https://www.github.com/kyubyong/transformer
"""
from __future__ import print_function

import numpy as np
import codecs
import regex
import random
import torch


def load_de_vocab(min_cnt):
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab(min_cnt):
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(src_sents, tgt_sents, min_cnt, max_len):
    de2idx, idx2de = load_de_vocab(min_cnt)
    en2idx, idx2en = load_en_vocab(min_cnt)

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        x = [de2idx.get(word, 1) for word in (src_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (tgt_sent + u" </S>").split()]
        if max(len(x), len(y)) <= max_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(src_sent)
            Targets.append(tgt_sent)

    # Pad
    X = np.zeros([len(x_list), max_len], np.int32)
    Y = np.zeros([len(y_list), max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, max_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, max_len - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data(train_src, train_tgt, min_cnt, max_len):
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(train_src, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(train_tgt, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents, min_cnt, max_len)
    return X, Y


def load_test_data(test_src, test_tgt, min_cnt, max_len):
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(test_src, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(test_tgt, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents, min_cnt, max_len)
    return X, Y, Sources, Targets  # (1064, 150)


def get_batch_indices(total_length, batch_size):
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index: current_index + batch_size], current_index
