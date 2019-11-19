import numpy as np

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch import nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS

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

    def __init__( self, Y, Y_time, H_dim=3, jitter=1e-3, random_kernel=False, init_param=False ):

        super(dmn_toy, self).__init__()

        self.set_data(Y=Y, Y_time=Y_time, H_dim=H_dim )

        self.jitter = jitter

        self.random_kernel = random_kernel

        self.weighted = False
        self.directed = False
        self.coord = True
        self.socpop = False

        # Covariance function (kernel) of the GP is defined here
        # self.kernel = pydmn.kernels.RBF( variance=torch.tensor([1.]), lengthscale=torch.tensor([12.]), random_param=random_kernel )
        self.kernel = pydmn.kernels.RBF( random_param=random_kernel )

        Kff = self.kernel(self.Y_time.reshape(-1,1)).detach()
        Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
        self.Lff_0 = Kff.cholesky() # cholesky lower triangular


        ### Variational Parameters ###
        # self.gp_mean_mu = torch.zeros((self.V_net,self.H_dim))
        # self.gp_mean_sigma = torch.ones((self.V_net,self.H_dim))
        self.gp_loc = torch.zeros((self.V_net,self.H_dim,self.T_net))
        self.gp_cov_tril = self.Lff_0.expand((self.V_net,self.H_dim,self.T_net,self.T_net))

        if self.random_kernel:
            self.kernel_param = torch.ones((2,2))

        if init_param:
            self.init_guide()
            self.gp_loc_init = self.gp_coord_mcmc[:,:,:,self.gp_coord_mcmc.shape[3]-1]
        else:
            self.gp_loc_init = torch.zeros((self.V_net,self.H_dim,self.T_net))

    def model(self):

        ## Optimization on Kernel params (often overfits)
        # pyro.module("kernel", self.kernel)
        kernel_model = self.kernel

        # Prior on Kernel params
        # priors = { 'lengthscale': dist.InverseGamma(torch.tensor([3.]),torch.tensor([36.])),
        #             'variance': dist.InverseGamma(torch.tensor([5.]),torch.tensor([5.])) }
        # lifted_module = pyro.random_module("kernel", self.kernel, priors)

        # /data/dupontslark/carmona/lib/python3.7/site-packages/pyro/primitives.py:347: FutureWarning: The `random_module` primitive is deprecated, and will be removed in a future release. Use `pyro.nn.Module` to create Bayesian modules from `torch.nn.Module` instances.
        # "modules from `torch.nn.Module` instances.", FutureWarning)
        # lifted_module = pyro.nn.PyroModule("kernel", self.kernel, priors)
        # pyro.nn.PyroModule(self.kernel)
        # pyro.nn.PyroModule[]
        # kernel_model = lifted_module()

        # x = torch.linspace(0,2,101)[1:]
        # f=dist.InverseGamma(torch.tensor([5.]),torch.tensor([5.]))
        # plt.plot(x,f.log_prob(x).exp())

        # Covariance matrix of observed times entailed by our kernel
        Kff = kernel_model(self.Y_time.reshape(-1,1))
        Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
        Lff = Kff.cholesky() # cholesky lower triangular

        # Mean function of the GP
        # gp_mean = torch.zeros((self.V_net,self.H_dim))

        # Latent coordinates
        # gp_coord_demean = torch.zeros( (self.V_net,self.H_dim,self.T_net) )
        gp_coord = torch.zeros( (self.V_net,self.H_dim,self.T_net) )

        zero_loc = torch.zeros( (self.T_net) )

        with pyro.plate('latent_coord_all', self.V_net*self.H_dim ):
            gp_coord = pyro.sample( f"latent_coord",
                                        dist.MultivariateNormal( zero_loc.expand(self.V_net * self.H_dim, self.T_net),
                                                                scale_tril=Lff ) )

        gp_coord = gp_coord.reshape(self.V_net, self.H_dim, self.T_net)
        # for v in range( self.V_net ):
        #     for h in range( self.H_dim ):
        # for v in pyro.plate( "vertex_loop", self.V_net ):
        #     for h in pyro.plate( f"H_dim_loop_{v}", self.H_dim ):
                # v=0; h=0
                # Sample mean coordinates #
                # gp_mean[v,h] = pyro.sample( f"gp_mean_{v}_{h}",
                #                             dist.Normal(loc=0,scale=1) )
                # Sample coordinates #
                # gp_coord[v,h,:] = pyro.sample( f"latent_coord_{v}_{h}",
                #                             dist.MultivariateNormal( zero_loc,
                #                                                     scale_tril=Lff ) )
                # Latent coordinates with mean
                # gp_coord[v,h,:] = gp_mean[v,h] + gp_coord_demean[v,h,:]
                # plt.plot( self.Y_time,gp_coord[v,h,:]); plt.hlines(y=self.gp_mean[v,h], xmin=self.Y_time.min(), xmax=self.Y_time.max() )

        ### Linear Predictor ###
        Y_linpred = torch.einsum('vht,wht->vwt', gp_coord, gp_coord)

        ### Link probability ###
        Y_link_prob = torch.sigmoid(Y_linpred)
        Y_link_prob_valid = Y_link_prob.flatten()[self.Y_valid_id.flatten()==1]
        # plt.scatter(self.Y_valid_obs,Y_link_prob_valid.detach())

        with pyro.plate("data", self.Y_valid_obs.shape[0]):
            pyro.sample("obs", dist.Bernoulli(Y_link_prob_valid).to_event(1), obs=self.Y_valid_obs)


    def guide(self):

        if self.random_kernel:
            self.kernel_param = pyro.param("kernel_param", torch.ones((2,2)), constraint=constraints.positive)
            pyro.sample( "lengthscale", dist.InverseGamma( self.kernel_param[0,0], self.kernel_param[0,1] ) )
            pyro.sample( "variance", dist.InverseGamma( self.kernel_param[1,0], self.kernel_param[1,1] ) )

        self.gp_loc = pyro.param( f"gp_loc", self.gp_loc_init )

        for v in range( self.V_net ):
            for h in range( self.H_dim ):
                # v=0; h=0
                # self.gp_mean_mu[v,h] = pyro.param( f"gp_mean_mu_{v}_{h}", torch.zeros(1) )
                # self.gp_mean_sigma[v,h] = pyro.param( f"gp_mean_sigma_{v}_{h}", torch.ones(1),
                #                                                 constraint=constraints.positive )
                # Location of the GP which will be infered
                # self.gp_loc[v,h,:] = pyro.param( f"gp_loc_{v}_{h}", self.gp_loc_init[v,h,:] )
                # Covariance function of the GP
                self.gp_cov_tril[v,h,:,:] = pyro.param( f"gp_cov_tril_{v}_{h}", self.Lff_0,
                                                constraint=constraints.lower_cholesky )

        with pyro.plate("latent_coord_all", self.V_net*self.H_dim ):
            pyro.sample( f"latent_coord",
                                    dist.MultivariateNormal( self.gp_loc.reshape(self.V_net * self.H_dim, self.T_net),
                                                            # scale_tril=self.Lff ) )
                                                            scale_tril=self.gp_cov_tril.reshape(self.V_net * self.H_dim, self.T_net, self.T_net) ) )
        # for v in range( self.V_net ):
        #     for h in range( self.H_dim ):
        # for v in pyro.plate( "vertex_loop", self.V_net ):
        #     for h in pyro.plate( f"H_dim_loop_{v}", self.H_dim ):
                # v=0; h=0
                # torch.mm(self.gp_cov_tril[v,h,:,:],self.gp_cov_tril[v,h,:,:].t())
                # pyro.sample( f"gp_mean_{v}_{h}",
                #                 dist.Normal(loc=self.gp_mean_mu[v,h], scale=self.gp_mean_sigma[v,h]) )
                # pyro.sample( f"latent_coord_{v}_{h}",
                #                         dist.MultivariateNormal( self.gp_loc[v,h,:],
                #                                                 # scale_tril=self.Lff ) )
                #                                                 scale_tril=self.gp_cov_tril[v,h,:,:] ) )


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

        # identifies elements of Y which will be considered for the likelihood
        self.Y_valid_id = np.array([np.tril( np.ones( (self.V_net,self.V_net) ), -1) for t in range(self.T_net)]).transpose((1,2,0))

        self.Y_valid_obs = self.Y.flatten()[self.Y_valid_id.flatten()==1]

        self.Y_link = torch.where( Y!=0, torch.ones_like(Y), torch.zeros_like(Y) )

        self.K_net = Y.shape[3] if len(Y.shape)==4 else 1

        assert self.T_net == Y_time.shape[0]
        self.Y_time = Y_time

        assert int(H_dim)>=1
        self.H_dim = int(H_dim)


    def gen_synth_net(self):

        # Mean function of the GP
        gp_mean = torch.zeros((self.V_net,self.H_dim))

        # Latent coordinates
        gp_coord_demean = torch.zeros( (self.V_net,self.H_dim,self.T_net) )
        gp_coord = torch.zeros( (self.V_net,self.H_dim,self.T_net) )

        # zero_loc = torch.zeros( (self.T_net) )
        for v in range( self.V_net ):
            for h in range( self.H_dim ):
                # v=0; h=0
                # Sample mean coordinates #
                # gp_mean[v,h] = pyro.sample( f"gp_mean_{v}_{h}",
                #                             dist.Normal(loc=0,scale=1) )
                # Sample coordinates #
                gp_coord[v,h,:] = dist.MultivariateNormal( self.gp_loc[v,h,:].detach(),
                                                                    scale_tril=self.gp_cov_tril[v,h,:,:].detach() ).sample()
                # Latent coordinates with mean
                # gp_coord[v,h,:] = gp_mean[v,h] + gp_coord_demean[v,h,:]

                # plt.plot( self.Y_time,gp_coord[v,h,:]); plt.hlines(y=self.gp_mean[v,h], xmin=self.Y_time.min(), xmax=self.Y_time.max() )

        ### Linear Predictors ###
        # Y_linpred = torch.zeros( (self.V_net, self.V_net, self.T_net) )
        Y_linpred = torch.einsum('vht,wht->vwt', gp_coord, gp_coord)

        ### Link probability ###
        Y_link_prob = torch.sigmoid(Y_linpred)

        Y_synth = torch.zeros_like(self.Y)
        Y_synth.flatten()[self.Y_valid_id.flatten()==1] = pyro.sample( "obs", dist.Bernoulli(Y_link_prob.flatten()[self.Y_valid_id.flatten()==1]).to_event(1) )
        return( Y_synth , gp_coord )

    def init_guide(self):
        nuts_kernel = NUTS(self.model)

        mcmc = MCMC(nuts_kernel, num_samples=1, warmup_steps=30)
        mcmc.run()

        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        # Get latent coordinates from mcmc #
        gp_coord_mcmc = np.array([[hmc_samples[f'latent_coord_{v}_{h}'] for h in range(self.H_dim)] for v in range(self.V_net)]).transpose(0,1,3,2)
        self.gp_coord_mcmc = torch.from_numpy(gp_coord_mcmc)
        # # Get probabilities from mcmc #
        # Y_linpred_mcmc = torch.einsum('vhtm,whtm->vwtm', gp_coord_mcmc, gp_coord_mcmc)
        # Y_link_prob_mcmc = torch.sigmoid(Y_linpred_mcmc)
