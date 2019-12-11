import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS

import PyDMN as pydmn

# define a PyTorch module for the DMN
class dmn_toy2( pyro.nn.PyroModule ):
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

    def __init__( self, edgelist, directed=False, weighted=False, coord = True, socpop = False, H_dim=3, sigma_k_prior_param=None, random_kernel=False, jitter=1e-3, init_mcmc=False ):

        super(dmn_toy2, self).__init__()

        self.directed = directed
        self.weighted = weighted
        self.coord = coord
        self.socpop = socpop

        self.edgelist = edgelist
        Y, [all_nodes, Y_time, all_layers] = pydmn.util.edgelist_to_tensor( edgelist = self.edgelist )

        Y = torch.tensor(Y)
        Y_time = torch.tensor(Y_time)
        self.all_layers = all_layers
        self.all_nodes = all_nodes

        self.set_data(Y=Y, Y_time=Y_time, H_dim=H_dim )

        self.random_kernel = random_kernel
        self.jitter = jitter

        # Number of directions to be considered
        # 2 if the network is directed: Send & receive
        # 1 if not
        if self.directed:
            self.n_dir = 2
        else:
            self.n_dir = 1


        # We consider one set of GPs for the propensity of connection
        # and (for weighted networks) one more set for the strenght of that connection
        if self.weighted:
            self.n_w = 2
            self.sigma_k_prior_param = torch.tensor([11.,20.]) if sigma_k_prior_param is None else sigma_k_prior_param
        else:
            self.n_w = 1

        ### Variational Parameters ###
        # To be set inside the guide
        self.gp_coord_mean_loc = None
        self.gp_coord_mean_scale = None
        self.gp_coord_demean = None
        self.gp_cov_tril = None

        if self.random_kernel:
            # If we optimise the parameters in the kernel, they will be stored here
            self.kernel_param = torch.ones((2,2))

        # If the Kernel IS random, we use PyroSample
        if self.random_kernel:
            self.kernel = pydmn.kernels.RBF()
            # self.kernel.lengthscale = pyro.nn.PyroSample( dist.InverseGamma(torch.tensor([51.]),torch.tensor([100.])) )
            # self.kernel.variance = pyro.nn.PyroSample( dist.InverseGamma(torch.tensor([51.]),torch.tensor([50.])) )

            # # Visualize InverseGamma
            # import matplotlib.pyplot as plt
            # x = torch.linspace(0.,5.,101)[1:]
            # plt.plot(x,dist.InverseGamma(torch.tensor([51.]),torch.tensor([100.])).log_prob(x).exp())
            # plt.plot(x,dist.Gamma(torch.tensor([50.]),torch.tensor([50.])).log_prob(x).exp())

        # Initial GP covariance, just to initialize random coordinates
        Kff = pydmn.kernels.RBF()(self.Y_time.reshape(-1,1)).detach()
        Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
        self.Lff_ini = Kff.cholesky() # cholesky lower triangular

        if init_mcmc:
            self.gp_system_mean_ini, self.gp_system_demean_ini, self.gp_coord_mean_ini, self.gp_coord_demean_ini = self.init_guide()
        else:
            self.gp_system_mean_ini = torch.zeros( (self.K_net,self.n_w) )
            self.gp_system_demean_ini = torch.zeros( (self.K_net,self.n_w,self.T_net) )
            self.gp_coord_mean_ini = torch.zeros( (self.V_net,self.H_dim,self.K_net,self.n_dir,self.n_w) )
            self.gp_coord_demean_ini = torch.zeros( (self.V_net,self.H_dim,self.K_net,self.n_dir,self.n_w,self.T_net) )

    def model(self):

        # If the Kernel IS NOT random, we declare the kernel within the model
        # if (not self.random_kernel):
        #     self.kernel = pydmn.kernels.RBF()

        # Covariance matrix of observed times entailed by our kernel
        # Kff = self.kernel(self.Y_time.reshape(-1,1))
        # Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
        # Lff = Kff.cholesky() # cholesky lower triangular
        Lff = self.Lff_ini

        # Sampling system-wide connectivity and average weights #
        with pyro.plate('gp_system_all', self.K_net*self.n_w ):
            # Mean function of the GPs
            gp_system_mean = pyro.sample( "gp_system_mean",
                                    dist.Normal( torch.zeros( (self.K_net*self.n_w) ),
                                                torch.tensor([0.1]) ) )
            # Demeaned GPs
            gp_system_demean = pyro.sample( "gp_system_demean",
                                            dist.MultivariateNormal( torch.zeros( (self.K_net*self.n_w, self.T_net) ),
                                                                        scale_tril=Lff ) )
        gp_system_mean = gp_system_mean.reshape(self.K_net,self.n_w)
        gp_system_demean = gp_system_demean.reshape(self.K_net, self.n_w, self.T_net)
        # Latent systemic evolution
        gp_system = gp_system_mean.expand(self.T_net, self.K_net, self.n_w).permute(1,2,0) + gp_system_demean

        # Sampling latent coordinates #
        with pyro.plate('gp_coord_all', self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w ):
            # Mean function of the GPs
            gp_coord_mean = pyro.sample( "gp_coord_mean",
                                    dist.Normal( torch.zeros( (self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w) ),
                                                torch.tensor([0.1]) ) )
            # Demeaned GPs
            gp_coord_demean = pyro.sample( "gp_coord_demean",
                                            dist.MultivariateNormal( torch.zeros( (self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w, self.T_net) ),
                                                                        scale_tril=Lff ) )

        gp_coord_mean = gp_coord_mean.reshape(self.V_net,self.H_dim,self.K_net,self.n_dir,self.n_w)
        gp_coord_demean = gp_coord_demean.reshape(self.V_net, self.H_dim, self.K_net, self.n_dir, self.n_w, self.T_net)

        # Latent coordinates
        gp_coord = gp_coord_mean.expand(self.T_net, self.V_net, self.H_dim, self.K_net, self.n_dir, self.n_w).permute(1,2,3,4,5,0) + gp_coord_demean

        ### Linear Predictor ###
        # Systemic component
        Y_linpred = gp_system.expand(self.V_net, self.V_net, self.K_net, self.n_w, self.T_net).permute(0,1,4,2,3)
        # distance between agents
        Y_linpred = Y_linpred + torch.einsum('uhkwt,vhkwt->uvtkw', gp_coord[:,:,:,0,:,:], gp_coord[:,:,:,self.n_dir-1,:,:])

        ### Link propensity (probability of occur) ###
        Y_link_prob = torch.sigmoid(Y_linpred[:,:,:,:,0])
        Y_link_prob_valid = Y_link_prob.flatten()[self.Y_valid_id.flatten()==1]

        with pyro.plate( "data", Y_link_prob_valid.shape[0]):
            pyro.sample( "obs", dist.Bernoulli(Y_link_prob_valid), obs=self.Y_link.flatten()[self.cond_Y_link] )

        ### Link expected weight (weight given occurrence) ###
        if self.weighted:
            with pyro.plate( "sigma_k_ind", self.K_net):
                sigma_k = pyro.sample( 'sigma_k', dist.InverseGamma( self.sigma_k_prior_param[0].expand(self.K_net), self.sigma_k_prior_param[1].expand(self.K_net) ) )
            Y_link_SDw = sigma_k.expand(self.V_net,self.V_net,self.T_net,self.K_net)

            Y_link_Ew = Y_linpred[:,:,:,:,1]
            # cond_Y_w: condition of being positive and valid weights (defined in set_data())
            Y_link_Ew_valid = Y_link_Ew.flatten()[self.cond_Y_w]
            Y_link_SDw_valid = Y_link_SDw.flatten()[self.cond_Y_w]
            with pyro.plate( "data_w", Y_link_Ew_valid.shape[0] ):
                pyro.sample( "obs_w", dist.Normal(Y_link_Ew_valid,Y_link_SDw_valid), obs=self.Y.flatten()[self.cond_Y_w] )

    def guide(self):

        # Posterior Covariance of the GP
        # if self.random_kernel:
        #     self.kernel_param = pyro.param("kernel_param", 50*torch.ones((2,2)), constraint=constraints.positive)
        #     pyro.sample( "kernel.lengthscale", dist.InverseGamma( self.kernel_param[0,0], self.kernel_param[0,1] ) )
        #     pyro.sample( "kernel.variance", dist.InverseGamma( self.kernel_param[1,0], self.kernel_param[1,1] ) )

        # Sampling Systemic components #
        self.gp_system_mean_loc = pyro.param("gp_system_mean_loc", self.gp_system_mean_ini )
        self.gp_system_mean_scale = pyro.param("gp_system_mean_scale", 0.1*torch.ones((self.K_net,self.n_w)), constraint=constraints.positive)
        self.gp_system_demean = pyro.param( f"gp_system_demean_loc", self.gp_system_demean_ini )
        # Posterior Covariance of the GP
        self.gp_system_cov_tril = pyro.param( "gp_system_cov_tril", self.Lff_ini.expand(self.K_net,self.n_w,self.T_net,self.T_net),
                                        constraint=constraints.lower_cholesky )
        with pyro.plate('gp_system_all', self.K_net*self.n_w ):
            # Posterior GP (mean function params) #
            pyro.sample( "gp_system_mean", dist.Normal( self.gp_system_mean_loc.reshape(self.K_net*self.n_w),
                                                self.gp_system_mean_scale.reshape(self.K_net*self.n_w) ) )
            # Posterior GP (demeaned) #
            pyro.sample( f"gp_system_demean",
                                    dist.MultivariateNormal( self.gp_system_demean.reshape(self.K_net*self.n_w , self.T_net),
                                                            scale_tril=self.gp_system_cov_tril.reshape(self.K_net*self.n_w , self.T_net, self.T_net) ) )

        # Sampling coordinates #
        self.gp_coord_mean_loc = pyro.param("gp_coord_mean_loc", self.gp_coord_mean_ini )
        self.gp_coord_mean_scale = pyro.param("gp_coord_mean_scale", 0.1*torch.ones((self.V_net,self.H_dim,self.K_net,self.n_dir,self.n_w)), constraint=constraints.positive)
        self.gp_coord_demean = pyro.param( f"gp_coord_demean_loc", self.gp_coord_demean_ini )
        # Posterior Covariance of the GP
        self.gp_cov_tril = pyro.param( "gp_cov_tril", self.Lff_ini.expand(self.V_net,self.H_dim,self.K_net,self.n_dir,self.n_w,self.T_net,self.T_net),
                                        constraint=constraints.lower_cholesky )
        with pyro.plate('gp_coord_all', self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w ):
            # Posterior GP (mean function params) #
            pyro.sample( "gp_coord_mean", dist.Normal( self.gp_coord_mean_loc.reshape(self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w),
                                                self.gp_coord_mean_scale.reshape(self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w) ) )
            # Posterior GP (demeaned) #
            pyro.sample( f"gp_coord_demean",
                                    dist.MultivariateNormal( self.gp_coord_demean.reshape(self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w , self.T_net),
                                                            scale_tril=self.gp_cov_tril.reshape(self.V_net*self.H_dim*self.K_net*self.n_dir*self.n_w , self.T_net, self.T_net) ) )

        # pyro.sample( "kernel.variance", dist.InverseGamma( self.kernel_param[1,0], self.kernel_param[1,1] ) )

        # Sampling variance of weights
        if self.weighted:
            self.sigma_k_post_loc = pyro.param("sigma_k_post_loc", torch.ones([1]), constraint=constraints.positive )
            self.sigma_k_post_scale = pyro.param("sigma_k_post_scale", torch.ones([1]), constraint=constraints.positive )
            with pyro.plate( "sigma_k_ind", self.K_net):
                sigma_k = pyro.sample( 'sigma_k', dist.InverseGamma( self.sigma_k_post_loc, self.sigma_k_post_scale ) )


    def set_data(self, Y, Y_time, H_dim=3):
        """
        Set data for dmn models as part of the nn.module
        and perform basic checks of data in DMN model
        """

        assert (len(Y.shape)>=3) and (len(Y.shape)<=4)

        # square adjacence matrices
        assert Y.shape[0]==Y.shape[1]

        self.Y = Y
        self.V_net, self.T_net, self.K_net = self.Y.shape[0], self.Y.shape[2], self.Y.shape[3]

        # identifies elements of Y which will be considered for the likelihood
        if self.directed:
            aux=torch.eye(self.V_net).expand(self.K_net,self.T_net,self.V_net,self.V_net).permute(3,2,1,0)
            self.Y_valid_id = torch.where( aux==0, torch.ones_like(Y), torch.zeros_like(Y) )
        else:
            self.Y_valid_id = torch.tril(torch.ones((self.V_net,self.V_net)),-1).expand(self.K_net,self.T_net,self.V_net,self.V_net).permute(3,2,1,0)

        self.Y_link = torch.where( Y!=0, torch.ones_like(Y), torch.zeros_like(Y) )
        self.cond_Y_link = self.Y_valid_id.flatten()==1

        if self.weighted:
            self.cond_Y_w = (self.Y_valid_id.flatten()==1)&(self.Y_link.flatten()==1)

        self.K_net = Y.shape[3] if len(Y.shape)==4 else 1

        assert self.T_net == Y_time.shape[0]
        self.Y_time = Y_time

        assert int(H_dim)>=1
        self.H_dim = int(H_dim)


    def gen_synth_net(self):

        # Mean function of the GP
        gp_coord_mean = torch.zeros((self.V_net,self.H_dim))

        # Latent coordinates
        gp_coord_demean = torch.zeros( (self.V_net,self.H_dim,self.T_net) )
        gp_coord = torch.zeros( (self.V_net,self.H_dim,self.T_net) )

        # zero_loc = torch.zeros( (self.T_net) )
        for v in range( self.V_net ):
            for h in range( self.H_dim ):
                # v=0; h=0
                # Sample mean coordinates #
                # gp_coord_mean[v,h] = pyro.sample( f"gp_coord_mean_{v}_{h}",
                #                             dist.Normal(loc=0,scale=1) )
                # Sample coordinates #
                gp_coord[v,h,:] = dist.MultivariateNormal( self.gp_loc[v,h,:].detach(),
                                                                    scale_tril=self.gp_cov_tril[v,h,:,:].detach() ).sample()
                # Latent coordinates with mean
                # gp_coord[v,h,:] = gp_coord_mean[v,h] + gp_coord_demean[v,h,:]

                # plt.plot( self.Y_time,gp_coord[v,h,:]); plt.hlines(y=self.gp_coord_mean[v,h], xmin=self.Y_time.min(), xmax=self.Y_time.max() )

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

        num_samples=3
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=7)
        mcmc.run()

        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        ### Get samples from mcmc ###
        # Systemic components
        gp_system_mean = torch.tensor(hmc_samples['gp_system_mean']).reshape(num_samples, self.K_net, self.n_w)
        gp_system_demean = torch.tensor(hmc_samples[f'gp_system_demean']).reshape(num_samples, self.K_net, self.n_w, self.T_net)
        # latent coordinates #
        gp_coord_mean = torch.tensor(hmc_samples['gp_coord_mean']).reshape(num_samples, self.V_net, self.H_dim, self.K_net, self.n_dir, self.n_w)
        gp_coord_demean = torch.tensor(hmc_samples[f'gp_coord_demean']).reshape(num_samples, self.V_net, self.H_dim, self.K_net, self.n_dir, self.n_w, self.T_net)

        # keep only last sample
        gp_system_mean_ini = gp_system_mean[num_samples-1]
        gp_system_demean_ini = gp_system_demean[num_samples-1]
        gp_coord_mean_ini = gp_coord_mean[num_samples-1]
        gp_coord_demean_ini = gp_coord_demean[num_samples-1]

        return (gp_system_mean_ini, gp_system_demean_ini, gp_coord_mean_ini,gp_coord_demean_ini)
        # gp_coord = gp_coord_mean.expand(self.T_net, self.V_net, self.H_dim, self.n_dir).permute(1,2,3,0) + gp_coord_demean

        # # Get probabilities from mcmc #
        # Y_linpred_mcmc = torch.einsum('vhtm,whtm->vwtm', gp_coord_mcmc, gp_coord_mcmc)
        # Y_link_prob_mcmc = torch.sigmoid(Y_linpred_mcmc)
