# -*- coding: utf-8 -*-
"""
Copied from https://www.github.com/kyubyong/transformer.
"""

from __future__ import print_function
import os
import regex
import codecs
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='Preprocessing')
parser.add_argument('--train_src', required=True, type=str, help='Path to training source data')
parser.add_argument('--train_tgt', required=True, type=str, help='Path to training target data')


def make_vocab(file_path, file_name):
    """Constructs vocabulary.
    Args:
      file_path: A string. Input file path.
      file_name: A string. Output file name.
    Writes vocabulary line by line to `preprocessed/file_name`
    """
    text = codecs.open(file_path, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    os.makedirs('preprocessed', exist_ok=True)
    with codecs.open('preprocessed/{}'.format(file_name), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


def main():
    args = parser.parse_args()
    print(args)
    make_vocab(args.train_src, "de.vocab.tsv")
    make_vocab(args.train_tgt, "en.vocab.tsv")
    print('Terminated')


if __name__ == '__main__':
    main()
