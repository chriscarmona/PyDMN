import numbers

from pyro.contrib.gp.parameterized import Parameterized


class Mean(Parameterized):
    """
    Base class for mean functions used in the Gaussian Process module.

    Every inherited class should implement a :meth:`forward` pass which takes input
    :math:`X` and returns their expected function f(X).

    References:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param int input_dim: Number of feature dimensions of inputs.
    :param torch.Tensor variance: Variance parameter of this kernel.
    """

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, X):
        r"""
        Calculates mean function vector of the inputs.

        :param torch.Tensor X: A 2D tensor with shape :math:`N \times input\_dim`.
        :returns: mean function vector of :math:`X`
            with shape :math:`N \times 1`
        """
        raise NotImplementedError
