import numpy as np

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch import nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
import pyro.contrib.gp as gp
from pyro.contrib.gp.parameterized import Parameterized
from pyro.contrib.gp.util import conditional
from pyro.distributions.util import eye_like

import PyDMN as pydmn

# define a PyTorch module for the DMN
class dmn_toy( nn.Module ):
    # this is equivalent to
    # pyro.contrib.gp.models.model.GPModel
    r"""
    Base class for Dynamic Networks using the Latent Space representation.

    Each node :math:`i` has four Gaussian Processes associated.
    1. Location in the latent social space for link propensity (probability of being connected)
    2. Location in the latent social space for edge weight (size of the connection)
    3. Sociability of the node
    3. Popularity of the node

    There are two ways to train the DMN model:

    + Using an MCMC algorithm
    + Using a variational inference on the pair :meth:`model`, :meth:`guide`:

    """

    def __init__( self, Y, Y_time, H_dim=3, jitter=1e-6 ):

        super(dmn_toy, self).__init__()

        self.set_data(Y=Y, Y_time=Y_time, H_dim=H_dim )

        # identifies entries in Y which will be used for inference
        self.Y_valid_id = torch.tensor(np.array([np.tril(np.ones((5,5)),-1) for t in range(self.T_net)]))

        self.jitter = jitter

        self.weighted = False
        self.directed = False
        self.coord = True
        self.socpop = False

        # Covariance function (kernel) of the GP is defined here
        self.kernel = pydmn.kernels.RBF(variance=1., lengthscale=12.)
        # self.kernel(self.Y_time.reshape(-1,1))

        self.gp_loc = torch.zeros((self.V_net, self.H_dim, self.T_net))
        self.gp_cov_tril = torch.eye(self.T_net).expand((self.V_net,self.H_dim,self.T_net,self.T_net))

    def model(self):

        # Prior covariance of dynamic coordinates across time
        # entailed by the kernel
        N = self.Y_time.size(0)
        Kff = self.kernel(self.Y_time.reshape(-1,1))
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky() # cholesky lower triangular

        # gp_loc_aux = self.gp_loc.reshape(-1,self.T_net).transpose(0,1)
        # gp_cov = self.kernel( self.Y_time.reshape(-1,1) )

        # Latent coordinates
        gp_coord = torch.zeros( (self.V_net,self.H_dim,self.T_net) )

        # Mean function of the GP
        self.gp_mean = pyro.param( "gp_mean", torch.zeros((self.V_net,self.H_dim)) )

        # Sample coordinates #
        zero_loc = torch.zeros( (self.T_net) )
        for v in pyro.plate( "vertex_loop", self.V_net ):
            for h in pyro.plate( f"H_dim_loop_{v}", self.H_dim ):
                # v=0; h=0
                gp_coord[v,h,:] = pyro.sample( f"latent_coord_{v}_{h}",
                                            dist.MultivariateNormal( zero_loc, scale_tril=Lff) )
                gp_coord[v,h,:] += self.gp_mean[v,h]
                # plt.plot( self.Y_time,gp_coord[v,h,:]); plt.hlines(y=self.gp_mean[v,h], xmin=self.Y_time.min(), xmax=self.Y_time.max() )

        ### Linear Predictors ###
        Y_linpred = torch.zeros( (self.V_net, self.V_net, self.T_net) )

        # identifies elements of Y which will be considered for the likelihood
        Y_valid = np.array([np.tril( np.ones( (self.V_net,self.V_net) ), -1) for t in range(self.T_net)]).transpose((1,2,0))

        # L2 distance between dynamic coordinates #
        send = gp_coord.expand( self.V_net, self.V_net, self.H_dim, self.T_net ).transpose(0,1)
        receive = gp_coord.expand( self.V_net, self.V_net, self.H_dim, self.T_net )
        Y_linpred += torch.norm(send - receive,dim=2)

        Y_link_prob = torch.sigmoid(Y_linpred)

        Y_link_prob_valid = Y_link_prob.flatten()[Y_valid.flatten()==1]

        with pyro.plate("data", Y_link_prob_valid.shape[0]):
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(Y_link_prob_valid).to_event(1), obs=self.Y.flatten()[Y_valid.flatten()==1])


    def guide(self):

        # N = self.Y_time.size(0)
        # Kff = self.kernel(self.Y_time.reshape(-1,1))
        # Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        # Lff = Kff.cholesky() # cholesky lower triangular

        # Location of the GP which will be infered
        self.gp_loc = pyro.param( "gp_loc", torch.zeros((self.V_net, self.H_dim, self.T_net)) )
        # self.gp_cov_tril = torch.eye(self.T_net).expand((self.V_net,self.H_dim,self.T_net,self.T_net))

        for v in range( self.V_net ):
            for h in range( self.H_dim ):
                # Covariance function of the GP
                self.gp_cov_tril[v,h,:,:] = pyro.param( f"gp_cov_tril_{v}_{h}", torch.eye(self.T_net) ,
                                                constraint=constraints.lower_cholesky )

        for v in pyro.plate( "vertex_loop", self.V_net ):
            for h in pyro.plate( f"H_dim_loop_{v}", self.H_dim ):
                # v=0; h=0
                pyro.sample( f"latent_coord_{v}_{h}",
                                        dist.MultivariateNormal( self.gp_loc[v,h,:],
                                                                scale_tril=self.gp_cov_tril[v,h,:,:] ) )


    def set_data(self, Y, Y_time, H_dim=3):
        """
        Set data for dmn models as part of the nn.module
        and perform basic checks of data in DMN model
        """

        assert (len(Y.shape)>=3) and (len(Y.shape)<=4)

        # square adjacence matrices
        assert Y.shape[0]==Y.shape[1]

        self.Y = Y
        self.V_net, self.T_net = self.Y.shape[0], self.Y.shape[2]

        self.Y_link = torch.where( Y!=0, torch.ones_like(Y), torch.zeros_like(Y) )

        self.K_net = Y.shape[3] if len(Y.shape)==4 else 1

        assert self.T_net == Y_time.shape[0]
        self.Y_time = Y_time

        assert int(H_dim)>=1
        self.H_dim = int(H_dim)
