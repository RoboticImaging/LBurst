# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo feature extraction

import pdb
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn


class Trainer (nn.Module):
    """ Helper class to train a deep network.
        Overload this class `forward_backward` for your actual needs.

    Usage:
        train = Trainer(net, loader, loss, optimizer)
        for epoch in range(n_epochs):
            train()
    """
    def __init__(self, net, loader, loss, optimizer, writer):
        nn.Module.__init__(self)
        self.net = net
        self.loader = loader
        self.loss_func = loss
        self.optimizer = optimizer
        self.writer = writer
        self.iterations = 0
    def iscuda(self):
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k:self.todevice(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)):
            return [self.todevice(v)  for v in x]

        if self.iscuda():
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def __call__(self):
        self.net.train()

        stats = defaultdict(list)

        for iter, inputs in enumerate(tqdm(self.loader)):
            inputs = self.todevice(inputs)

            self.optimizer.zero_grad()

            loss, details = self.forward_backward(inputs)
            if torch.isnan(loss):
                raise RuntimeError('Loss is NaN')

            self.optimizer.step()

            for key, val in details.items():
                stats[key].append( val )
                self.writer.add_scalar(key, val, self.iterations)
            self.writer.add_scalar('loss', loss, self.iterations)
            self.iterations += 1

        print(" Summary of losses during this epoch:")
        mean = lambda lis: sum(lis) / len(lis) if len(lis) > 0 else 0
        for loss_name, vals in stats.items():
            N = 1 + len(vals)//10
            print(f"  - {loss_name:20}:", end='')
            print(f" {mean(vals[:N]):.6f} --> {mean(vals[-N:]):.6f} (avg: {mean(vals):.6f})")
        return mean(stats['loss'])

    def forward_backward(self, inputs):
        raise NotImplementedError()
