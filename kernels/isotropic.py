import torch
from torch.distributions import constraints
# from torch.nn import Parameter

import pyro
import pyro.distributions as dist

from .kernel import Kernel

class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension)

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """
    def __init__(self, variance=None, lengthscale=None, random_param=False):
        super(Isotropy, self).__init__()

        self.random_param = random_param

        # # Visualize InverseGamma
        # import matplotlib.pyplot as plt
        # x = torch.linspace(0.,3.,101)
        # plt.plot(x,dist.InverseGamma(torch.tensor([11.]),torch.tensor([10.])).log_prob(x).exp())
        # plt.plot(x,dist.Gamma(torch.tensor([10.]),torch.tensor([10.])).log_prob(x).exp())

        # Set lengthscale parameter
        if random_param:
            self.lengthscale = pyro.nn.PyroSample( dist.InverseGamma(torch.tensor([4.]),torch.tensor([30.])) )
        else:
            lengthscale = torch.tensor(10.) if lengthscale is None else lengthscale
            self.lengthscale = pyro.param("lengthscale", lengthscale, constraint=constraints.greater_than(1) )

        # Set variance parameter
        if random_param:
            self.variance = pyro.nn.PyroSample( dist.InverseGamma(torch.tensor([11.]),torch.tensor([10.])) )
        else:
            variance = torch.tensor(1.) if variance is None else variance
            self.variance = pyro.param("variance", variance, constraint=constraints.positive)


    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        if X.size != Z.size:
            raise ValueError("Inputs must have the same number of features.")

        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)


class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """
    def __init__(self, variance=None, lengthscale=None, random_param=False):
        super(RBF, self).__init__(variance=variance, lengthscale=lengthscale, random_param=random_param)

    def forward(self, X, Z=None):
        r2 = self._square_scaled_dist(X, Z)
        return self.variance * torch.exp(-0.5 * r2)
