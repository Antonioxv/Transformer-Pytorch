import torch
from utils import LabelSmoothing

from torch.autograd import Variable


def simple_compute_loss(logits, probs, y, num_tgt_vocab, use_cuda=True):
    # Cal
    _, preds = torch.max(logits, -1)
    # print(preds)
    istarget = (1. - y.eq(0.).float()).view(-1)
    acc = torch.sum(preds.eq(y).float().view(-1) * istarget) / torch.sum(istarget)

    # Loss
    if use_cuda:
        y_onehot = torch.zeros(logits.size()[0] * logits.size()[1], num_tgt_vocab).cuda()
    else:
        y_onehot = torch.zeros(logits.size()[0] * logits.size()[1], num_tgt_vocab)
    y_onehot = Variable(y_onehot.scatter_(1, y.view(-1, 1).data, 1))

    ls = LabelSmoothing()
    y_smoothed = ls(y_onehot)

    loss = - torch.sum(y_smoothed * torch.log(probs), dim=-1)
    mean_loss = torch.sum(loss * istarget) / torch.sum(istarget)

    return mean_loss, preds, acc
