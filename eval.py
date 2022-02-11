from __future__ import print_function

import os
import sys

import torch
import argparse
import codecs

import numpy as np

from loss import simple_compute_loss
from model import Transformer
from data import load_test_data, load_de_vocab, load_en_vocab
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# from nltk.translate.bleu_score import corpus_bleu

print(torch.__version__)
print(torch.cuda.is_available())
use_cuda = torch.cuda.is_available()

# Evaluate
parser = argparse.ArgumentParser(description='Evaluate')
# training params
parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment dir is needed for training')
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--result_dir', type=str, default='results')
parser.add_argument('--run_dir', type=str, default='runs')
parser.add_argument('--eval_epoch', type=int, required=True, help='Select a epoch to evaluate')
parser.add_argument('--batch_size', type=int, default=32)

# data loading params
parser.add_argument('--min_cnt', type=int, required=True, default=20,
                    help='Words whose occurred less than min_cnt are encoded as <UNK>')
parser.add_argument('--max_src_seq_len', type=int, default=10, help='Maximum number of words in a source sentence')
parser.add_argument('--max_tgt_seq_len', type=int, default=10, help='Maximum number of words in a target sentence')
parser.add_argument('--test_src', type=str, default='corpora/test_data/IWSLT16.TEDX.tst2014.de-en.de.xml', help='Test source file path')
parser.add_argument('--test_tgt', type=str, default='corpora/test_data/IWSLT16.TEDX.tst2014.de-en.en.xml', help='Test target file path')

# network params
parser.add_argument('--src_padding_idx', type=int, default=0)
parser.add_argument('--tgt_padding_idx', type=int, default=0)
parser.add_argument('--vocab_emb_dim', type=int, default=512)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--k_dim', type=int, default=64)
parser.add_argument('--v_dim', type=int, default=64)
parser.add_argument('--ff_dim', type=int, default=2048)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--dropout', type=float, default=0.1)


def main():
    # Args
    args = parser.parse_args()
    print(args)

    # Make dirs
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    model_dir = os.path.join(experiment_dir, args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    result_dir = os.path.join(experiment_dir, args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    # Tensorboard
    run_dir = os.path.join(experiment_dir, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Load model
    de2idx, idx2de = load_de_vocab(args.min_cnt)
    en2idx, idx2en = load_en_vocab(args.min_cnt)
    num_src_vocab = len(de2idx)
    num_tgt_vocab = len(en2idx)

    model = Transformer(num_src_vocab=num_src_vocab, num_tgt_vocab=num_tgt_vocab,
                        src_padding_idx=args.src_padding_idx, tgt_padding_idx=args.tgt_padding_idx,
                        vocab_emb_dim=args.vocab_emb_dim, max_seq_len=min(args.max_src_seq_len, args.max_tgt_seq_len),
                        model_dim=args.model_dim, k_dim=args.k_dim, v_dim=args.v_dim, ff_dim=args.ff_dim,
                        num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout)
    if use_cuda:
        model.load_state_dict(torch.load(model_dir + '/model_epoch_%02d' % args.eval_epoch + '.pth'))
        model.eval()
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_dir + '/model_epoch_%02d' % args.eval_epoch + '.pth', map_location="cpu"))
        model.eval()
    print('Model loaded.')

    # Load data
    max_len = min(args.max_src_seq_len, args.max_tgt_seq_len)
    X, Y, Sources, Targets = load_test_data(args.test_src, args.test_tgt, args.min_cnt, max_len)

    # Calculate total batch count
    num_batch = len(X) // args.batch_size

    # Evaluate
    with codecs.open(result_dir + '/model%d.txt' % args.eval_epoch, 'w', 'utf-8') as fout:
        list_of_refs, hypotheses = [], []
        for i in range(num_batch):
            # Get mini-batches
            x = X[i * args.batch_size: (i + 1) * args.batch_size]
            y = Y[i * args.batch_size: (i + 1) * args.batch_size]
            sources = Sources[i * args.batch_size: (i + 1) * args.batch_size]
            targets = Targets[i * args.batch_size: (i + 1) * args.batch_size]

            # Autoregressive inference
            if use_cuda:
                x_ = Variable(torch.LongTensor(x).cuda())
                preds_t = Variable(torch.LongTensor(y))
                # preds_t = torch.LongTensor(np.zeros((args.batch_size, max_len), np.int32)).cuda()
            else:
                x_ = Variable(torch.LongTensor(x))
                preds_t = Variable(torch.LongTensor(y))
                # preds_t = torch.LongTensor(np.zeros((args.batch_size, max_len), np.int32))

            preds = Variable(preds_t)
            for j in range(max_len):
                # Forward
                logits, probs = model(x_, preds)
                # Calculate loss, _preds and acc
                _, _preds, _ = simple_compute_loss(logits, probs, preds, num_tgt_vocab, use_cuda)
                # sys.exit(0)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())
            preds = preds.data.cpu().numpy()

            # Write to file
            for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                # print(got)

                fout.write("- source: " + source + "\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()

                # if len(ref) > 3 and len(hypothesis) > 0:
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)

            # # Calculate bleu score
            # score = corpus_bleu(list_of_refs, hypotheses)
            # fout.write("Bleu Score = " + str(100 * score))

    writer.close()
    print('Terminated')


if __name__ == '__main__':
    main()
