import pyro

class Kernel(pyro.nn.PyroModule):
    """
    Base class for kernels used in this Gaussian Process module.

    Every inherited class should implement a :meth:`forward` pass which takes inputs
    :math:`X`, :math:`Z` and returns their covariance matrix.

    To construct a new kernel from the old ones, we can use methods :meth:`add`,
    :meth:`mul`, :meth:`exp`, :meth:`warp`, :meth:`vertical_scale`.

    References:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    """

    def __init__(self):
        super(Kernel, self).__init__()


    def forward(self, X, Z=None):
        r"""
        Calculates covariance matrix of inputs on active dimensionals.

        :param torch.Tensor X: A 2D tensor with shape :math:`N \times input\_dim`.
        :param torch.Tensor Z: An (optional) 2D tensor with shape
            :math:`M \times input\_dim`.
        :param bool diag: A flag to decide if we want to return full covariance matrix
            or just its diagonal part.
        :returns: covariance matrix of :math:`X` and :math:`Z` with shape
            :math:`N \times M`
        :rtype: torch.Tensor
        """
        raise NotImplementedError
