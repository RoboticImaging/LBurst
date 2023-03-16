# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

# Modified for RoBLo features extraction

import pdb
import numpy as np
import torch
import torch.nn as nn


class APLoss (nn.Module):
    """ differentiable AP loss, through quantization.

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: list of query AP (for each n in {1..N})
                 Note: typically, you want to minimize 1 - mean(AP)
    """
    def __init__(self, nq=25, min=0, max=1, euc=False):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        self.euc = euc
        gap = max - min
        assert gap > 0

        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        a = (nq-1) / gap
        q.weight.data[:nq] = -a
        q.bias.data[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))
        q.weight.data[nq:] = a
        q.bias.data[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)
        q.weight.data[0] = q.weight.data[-1] = 0
        q.bias.data[0] = q.bias.data[-1] = 1

    def compute_AP(self, x, label):
        N, M = x.shape
        if self.euc:
            x = 1 - torch.sqrt(2.001 - 2*x)

        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:,:self.nq], q[:,self.nq:]).clamp(min=0)

        nbs = q.sum(dim=-1)
        rec = (q * label.view(N,1,M).float()).sum(dim=-1)
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))
        rec /= rec.sum(dim=-1).unsqueeze(1)

        ap = (prec * rec).sum(dim=-1)
        return ap

    def forward(self, x, label):
        assert x.shape == label.shape
        return self.compute_AP(x, label)
