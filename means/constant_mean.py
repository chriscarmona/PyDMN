#!/usr/bin/env python3

import torch
from .mean import Mean
from torch.nn import Parameter

class ConstantMean(Mean):

    def __init__(self, constant=None ):
        super(ConstantMean, self).__init__()

        constant = torch.zeros(1) if constant is None else constant
        self.constant = Parameter(constant)

    def forward(self, X):
        return self.constant.expand(X.size(0))
