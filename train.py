from __future__ import print_function

import os
import time
import torch
import pickle
import argparse

import torch.optim as optim

from model import Transformer
from loss import simple_compute_loss
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from data import load_de_vocab, load_en_vocab, load_train_data, get_batch_indices

print(torch.__version__)
print(torch.cuda.is_available())
use_cuda = torch.cuda.is_available()

# Train
parser = argparse.ArgumentParser(description='Train')
# training params
parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment dir is needed for training')
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--run_dir', type=str, default='runs')
parser.add_argument('--preload', type=int, default=None)  # Epoch of preloaded model for resuming training
parser.add_argument('--lr', type=float, required=True)  # default=0.0001
parser.add_argument('--max_epochs', type=int, required=True)  # default=20
parser.add_argument('--batch_size', type=int, default=32)

# data loading params
parser.add_argument('--min_cnt', type=int, required=True, default=20,
                    help='Words whose occurred less than min_cnt are encoded as <UNK>')
parser.add_argument('--max_src_seq_len', type=int, default=10, help='Maximum number of words in a source sentence')
parser.add_argument('--max_tgt_seq_len', type=int, default=10, help='Maximum number of words in a target sentence')
parser.add_argument('--train_src', type=str, default='corpora/train_data/train.tags.de-en.de', help='Train source file path')
parser.add_argument('--train_tgt', type=str, default='corpora/train_data/train.tags.de-en.en', help='Train target file path')

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
    # Tensorboard
    run_dir = os.path.join(experiment_dir, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Initialize model
    de2idx, idx2de = load_de_vocab(args.min_cnt)
    en2idx, idx2en = load_en_vocab(args.min_cnt)
    num_src_vocab = len(de2idx)
    num_tgt_vocab = len(en2idx)

    model = Transformer(num_src_vocab=num_src_vocab, num_tgt_vocab=num_tgt_vocab,
                        src_padding_idx=args.src_padding_idx, tgt_padding_idx=args.tgt_padding_idx,
                        vocab_emb_dim=args.vocab_emb_dim, max_seq_len=min(args.max_src_seq_len, args.max_tgt_seq_len),
                        model_dim=args.model_dim, k_dim=args.k_dim, v_dim=args.v_dim, ff_dim=args.ff_dim,
                        num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout)
    model.train()
    if use_cuda:
        model.cuda()
    # torch.backends.cudnn.benchmark = True
    print('Model initialized.')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    # Params
    start_epoch = 1
    history = {'current_batches': 0}

    # Preload
    if args.preload is not None:
        start_epoch = int(args.preload)
        if os.path.exists(model_dir + '/model_epoch_%02d.pth' % args.preload):
            model.load_state_dict(torch.load(model_dir + '/model_epoch_%02d.pth' % args.preload))
        if os.path.exists(model_dir + '/optimizer.pth'):
            optimizer.load_state_dict(torch.load(model_dir + '/optimizer.pth'))
        if os.path.exists(model_dir + '/history.pkl'):
            with open(model_dir + '/history.pkl') as in_file:
                history = pickle.load(in_file)

    current_batches = history['current_batches']

    # Load data
    max_len = min(args.max_src_seq_len, args.max_tgt_seq_len)
    X, Y = load_train_data(args.train_src, args.train_tgt, args.min_cnt, max_len)
    # Calculate total batch count
    num_batch = len(X) // args.batch_size

    # Train
    for epoch in range(start_epoch, args.max_epochs + 1):
        current_batch = 0
        for index, current_index in get_batch_indices(len(X), args.batch_size):
            # 1) Batch data loading
            tic = time.time()
            if use_cuda:
                x_batch = Variable(torch.LongTensor(X[index]).cuda())
                y_batch = Variable(torch.LongTensor(Y[index]).cuda())
            else:
                x_batch = Variable(torch.LongTensor(X[index]))
                y_batch = Variable(torch.LongTensor(Y[index]))
            toc = time.time()

            # 2) Optimize params
            tic_r = time.time()
            if use_cuda:
                torch.cuda.synchronize()

            # Forward
            optimizer.zero_grad()
            logits, probs = model(x_batch, y_batch)

            # Calculate loss and acc
            loss, _, acc = simple_compute_loss(logits, probs, y_batch, num_tgt_vocab, use_cuda)

            # Backward
            loss.backward()
            optimizer.step()

            if use_cuda:
                torch.cuda.synchronize()
            toc_r = time.time()

            # 3) Record
            current_batches += 1
            current_batch += 1
            if current_batches % 10 == 0:
                writer.add_scalar('Train/loss', loss.item(), global_step=current_batches)
                writer.add_scalar('Train/accuracy', acc.item(), global_step=current_batches)
            if current_batches % 5 == 0:
                print('epoch %d, batch %d/%d, loss %f, acc %f' % (epoch, current_batch, num_batch, loss.item(), acc.item()))
                print('batch loading used time %f, model forward used time %f' % (toc - tic, toc_r - tic_r))
            if current_batches % 100 == 0:
                writer.export_scalars_to_json(os.path.join(log_dir, 'all_scalars.json'))

        # Save history, model and optimizer
        with open(model_dir + '/history.pkl', 'wb') as out_file:
            pickle.dump(history, out_file)

        checkpoint_path = model_dir + '/model_epoch_%02d' % epoch + '.pth'
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), model_dir + '/optimizer.pth')

    writer.close()
    print('Terminated')


if __name__ == '__main__':
    main()
