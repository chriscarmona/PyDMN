#!/usr/bin/env python3

import torch
from .mean import Mean
from torch.nn import Parameter

class LinearMean(Mean):
    def __init__( self, input_dim, constant=None, slope=None ):
        super().__init__()

        self.input_dim = input_dim

        constant = torch.zeros(1) if constant is None else constant
        self.constant = Parameter(constant)

        slope = torch.zeros(input_dim) if slope is None else slope
        self.slope = Parameter(slope)

    def forward(self, X):
        return self.constant + X.view(-1,self.input_dim).matmul(self.slope.view(-1,1)).squeeze(-1)
