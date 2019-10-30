#!/usr/bin/env python3

import torch
from .mean import Mean


class ZeroMean(Mean):

    def __init__(self):
        super(ZeroMean, self).__init__()

    def forward(self, X):
        return torch.zeros(X.size(0), dtype=X.dtype, device=X.device)
