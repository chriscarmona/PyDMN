import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
import pyro.contrib.gp as gp
from pyro.contrib.gp.parameterized import Parameterized
from pyro.contrib.gp.util import conditional

import PyDMN

class dmn( Parameterized ):
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

    def __init__( self, Y, Y_time, H_dim=3, X=None,
                 weighted=True, directed=True, coord=False, socpop=True,
                 jitter=1e-6, whiten=False ):

        super(dmn, self).__init__()

        self.set_data(Y=Y, Y_time=Y_time, H_dim=H_dim, X=X )
        self.whiten = whiten

        self.jitter = jitter

        self.weighted = weighted
        self.directed = directed
        self.coord = coord
        self.socpop = socpop

        # Dimensions of objects change if the network is weighted and/or directed
        self.lw_dim = (2 if weighted else 1)
        self.sr_dim = (2 if directed else 1)

        self.kernel = Parameterized() # kernels modules will be added here
        self.gp = Parameterized() # GP modules will be added here

        if self.weighted:
            sigma_Y = torch.ones(self.K_net)
            self.sigma_Y = Parameter(sigma_Y)
            self.set_constraint("sigma_Y", constraints.positive)

        ### Latent locations ###
        if coord:
            # Kernels for GP of latent locations #
            # Assume that the kernel is the same for all agents, i.e. the dynamics in the latent space is the same #
            self.kernel.coord = Parameterized()
            self.kernel.coord.link = gp.kernels.RBF( input_dim = 1 )
            if self.weighted:
                self.kernel.coord.weight = gp.kernels.RBF( input_dim = 1 )

            ## Dynamic latent locations ##
            self.gp.coord = Parameterized()

            # GP location (without mean fun) #
            # dimensions: (V_net, H_dim, sr_dim, lw_dim, T_net) = (1 per agent, 1 per lat dim, send & rec, link & weight, Time )
            gp_coord_loc = torch.randn( self.V_net, self.H_dim, self.sr_dim, self.lw_dim, self.T_net )
            self.gp.coord.loc = Parameter( gp_coord_loc )

            # Mean function: constant #
            # dimensions: (V_net, H_dim, sr_dim, lw_dim) = (1 per agent, 1 per lat dim, send & rec, link & weight)
            gp_coord_loc_mean = torch.randn( self.V_net, self.H_dim, self.sr_dim, self.lw_dim )
            self.gp.coord.loc_mean = Parameter( gp_coord_loc_mean )

            # Lower Cholesky of the Covariance matrix of latent coordinates #
            # dimensions: (V_net, H_dim, sr_dim, lw_dim, T_net, T_net) = (1 per agent, 1 per lat dim, send & rec, link & weight, Time, Time )
            gp_coord_cov_tril_unconst = torch.diag_embed( torch.ones(self.V_net, self.H_dim, self.sr_dim, self.lw_dim, self.T_net) )
            self.gp.coord.cov_tril_unconst = Parameter(gp_coord_cov_tril_unconst)
            self.gp.coord.cov_tril = torch.stack([
                                        torch.stack([
                                            torch.stack([
                                                torch.stack([
                            transform_to( constraints.lower_cholesky )( self.gp.coord.cov_tril_unconst[v,h,sr_i,lw_i,:,:] )
                                                for lw_i in range(self.lw_dim)])
                                            for sr_i in range(self.sr_dim)])
                                        for h in range(self.H_dim)])
                                    for v in range(self.V_net)])
        else:
            self.kernel.coord = None
            self.gp.coord = None

        ### Sociability and Popularity ###
        if socpop:
            # Kernels for GP of Sociability and Popularity params #
            # Assume that the kernel is the same for sociability and popularity #
            self.kernel.socpop = Parameterized()
            self.kernel.socpop.link = gp.kernels.RBF( input_dim = 1 )
            if self.weighted:
                self.kernel.socpop.weight = gp.kernels.RBF( input_dim = 1 )

            ## Dynamic Sociability and Popularity ##
            self.gp.socpop = Parameterized()

            # dimensions: (V_net, 2, lw_dim, T_net) = (1 per agent, soc & pop, link & weight, Time )
            gp_socpop_loc = torch.randn( self.V_net, 2, self.lw_dim, self.T_net )
            self.gp.socpop.loc = Parameter( gp_socpop_loc )

            # Mean function: constant #
            gp_socpop_loc_mean = torch.randn( self.V_net, 2, self.lw_dim )
            self.gp.socpop.loc_mean = Parameter( gp_socpop_loc_mean )

            # GP.socpop.Cov_tril: Lower Cholesky of the Covariance matrix of latent socpopinates
            # dimensions: (V_net, 2, lw_dim, T_net, T_net) = (1 per agent, soc & pop, link & weight, Time, Time )
            gp_socpop_cov_tril_unconst = torch.diag_embed( torch.ones(self.V_net, 2, self.lw_dim, self.T_net) )
            self.gp.socpop.cov_tril_unconst = Parameter(gp_socpop_cov_tril_unconst)
            self.gp.socpop.cov_tril = torch.stack([
                                                    torch.stack([
                                                        torch.stack([
                            transform_to( constraints.lower_cholesky )( self.gp.socpop.cov_tril_unconst[v,sp_i,lw_i,:,:] )
                                                        for lw_i in range(self.lw_dim)])
                                                    for sp_i in range(2)])
                                        for v in range(self.V_net)])
        else:
            self.kernel.socpop = None
            self.gp.socpop = None

    # @autoname.scope(prefix="DMN") # generates error
    def model(self):
        self.set_mode("model")

        # Sample the coordinates
        # dimensions: (V_net, H_dim, sr_dim, lw_dim, T_net) = (1 per agent, 1 per lat dim, send & rec, link & weight, Time )
        if self.coord:

            # Calculates lower cholesky for all soc, pop
            Kff = [ eval('self.kernel.coord.'+['link','weight'][lw_i])(self.Y_time).contiguous() for lw_i in range(self.lw_dim)]
            for lw_i in range(self.lw_dim): Kff[lw_i].view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
            Lff_coord = [ Kff[lw_i].cholesky() for lw_i in range(self.lw_dim)]

            # Gaussian process for the sociability and popularity #
            gp_coord = torch.stack([
                            torch.stack([
                                torch.stack([
                                    torch.stack([
                                PyDMN.util.GP_sample( name=f'f_coord_v{v}_h{h}_sr{sr_i}_lw{lw_i}',
                                            X=self.Y_time,
                                            f_loc=self.gp.coord.loc[v,h,sr_i,lw_i,:],
                                            f_loc_mean=self.gp.coord.loc_mean[v,h,sr_i,lw_i],
                                            f_scale_tril=self.gp.coord.cov_tril[v,h,sr_i,lw_i,:,:],
                                            Lff=Lff_coord[lw_i],
                                            whiten=self.whiten )
                                    for lw_i in range(self.lw_dim)])
                                for sr_i in range(self.sr_dim)])
                            for h in range(self.H_dim)])
                        for v in range(self.V_net)])

        # Sample the Sociability and Popularity ##
        # dimensions: (V_net, 2, lw_dim, T_net) = (1 per agent, soc & pop, link & weight, Time )
        if self.socpop:

            # Calculates lower cholesky for all soc, pop
            Kff = [ eval('self.kernel.socpop.'+['link','weight'][lw_i])(self.Y_time).contiguous() for lw_i in range(self.lw_dim)]
            for lw_i in range(self.lw_dim): Kff[lw_i].view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
            Lff_socpop = [ Kff[lw_i].cholesky() for lw_i in range(self.lw_dim)]

            # Gaussian process for the sociability and popularity #
            gp_socpop = torch.stack([
                                    torch.stack([
                                        torch.stack([
                                PyDMN.util.GP_sample( name=f'f_socpop_v{v}_{["soc","pop"][sp_i]}_lw{lw_i}',
                                            X=self.Y_time,
                                            f_loc=self.gp.socpop.loc[v,sp_i,lw_i,:],
                                            f_loc_mean=self.gp.socpop.loc_mean[v,sp_i,lw_i],
                                            f_scale_tril=self.gp.socpop.cov_tril[v,sp_i,lw_i,:,:],
                                            Lff=Lff_socpop[lw_i],
                                            whiten=self.whiten )
                                        for lw_i in range(self.lw_dim)])
                                    for sp_i in range(2)])
                            for v in range(self.V_net)])


        ### Calculate Linear Predictors ###
        Y_linpred = torch.zeros( self.V_net, self.V_net, self.T_net, self.lw_dim )

        # identifies diagonal elements, which will not be considered when computing the likelihood
        Y_diag = torch.diag_embed( torch.ones(self.T_net, self.V_net) ).transpose(0,2).flatten()

        if self.coord:
            # gp_coord.shape # gp_coord[v,h,sr_i,lw_i,t]
            send = gp_coord[:,:,0,:,:].expand( self.V_net, self.V_net, self.H_dim, self.lw_dim, self.T_net)
            receive = gp_coord[:,:,1,:,:].expand( self.V_net, self.V_net, self.H_dim, self.lw_dim, self.T_net).transpose(0,1)
            Y_linpred += ((send - receive)**2).sum(dim=2).rsqrt().transpose(2,3)

        if self.socpop:
            soc = gp_socpop[:,0,:,:].expand( self.V_net, self.V_net, self.lw_dim, self.T_net)
            pop = gp_socpop[:,1,:,:].expand( self.V_net, self.V_net, self.lw_dim, self.T_net).transpose(0,1)
            Y_linpred += (soc + pop).transpose(2,3)

        ### Sampling 0-1 links ###
        Y_probs = torch.sigmoid(Y_linpred[:,:,:,0].flatten())[Y_diag==0]
        Y_dist_link = dist.Bernoulli( Y_probs )
        Y_dist_link = Y_dist_link.expand_by(Y_probs.shape[:-Y_probs.dim()]).to_event(Y_probs.dim())
        return pyro.sample("Y_link", Y_dist_link, obs=(self.Y_link.flatten())[Y_diag==0] )

        ### Sampling weights ###
        if self.weighted:
            Y_weighted_id = (Y_diag==0)&(self.Y.flatten()!=0)

            Y_loc = (Y_linpred[:,:,:,1].flatten())[Y_weighted_id]
            Y_scale = self.sigma_Y.expand(self.V_net, self.V_net, self.T_net).flatten()[Y_weighted_id]

            Y_dist = dist.Normal( Y_loc, Y_scale )
            Y_dist = Y_dist.expand_by(Y_loc.shape[:-Y_loc.dim()]).to_event(Y_loc.dim())

            pyro.sample( "Y", Y_dist, obs=(self.Y.flatten())[Y_weighted_id] )

    # @autoname.scope(prefix="DMN") # generates error
    def guide(self):
        self.set_mode("guide")

        if self.coord:
            for v in range(self.V_net):
                for h in range(self.H_dim):
                    for sr_i in range(self.sr_dim):
                        for lw_i in range(self.lw_dim):
                            pyro.sample( f'f_coord_v{v}_h{h}_sr{sr_i}_lw{lw_i}',
                                        dist.MultivariateNormal( self.gp.coord.loc[v,h,sr_i,lw_i,:],
                                                                scale_tril=self.gp.coord.cov_tril[v,h,sr_i,lw_i,:,:] )
                                                                .to_event(self.gp.coord.loc[v,h,sr_i,lw_i,:].dim()-1) )

        if self.socpop:
            for v in range(self.V_net):
                for sp_i in range(2):
                    for lw_i in range(self.lw_dim):
                        pyro.sample( f'f_socpop_v{v}_{["soc","pop"][sp_i]}_lw{lw_i}',
                                    dist.MultivariateNormal( self.gp.socpop.loc[v,sp_i,lw_i,:],
                                                            scale_tril=self.gp.socpop.cov_tril[v,sp_i,lw_i,:,:] )
                                                            .to_event(self.gp.socpop.loc[v,sp_i,lw_i,:].dim()-1) )

    def forward(self, Y_time_new, num_particles=30):
        r"""
        Computes something
        """
        self.set_mode("guide")
        # Y_time_new=torch.arange(0,30,0.25)

        ## Generate coordinates for new times##
        if self.coord:

            Lff_coord = []; Kfs_coord = []; Kss_coord=[]
            for lw_i in range(self.lw_dim):
                Kff = eval('self.kernel.coord.'+['link','weight'][lw_i])(self.Y_time).contiguous()
                Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
                Lff_coord.append( Kff.cholesky() )

                Kfs_coord.append( eval('self.kernel.coord.'+['link','weight'][lw_i])(self.Y_time, Y_time_new) )

                Kss_coord.append( eval('self.kernel.coord.'+['link','weight'][lw_i])(Y_time_new).contiguous() )
                Kss_coord[lw_i].view(-1)[::Kss_coord[lw_i].shape[0] + 1] += self.jitter  # add jitter to the diagonal

            gp_coord_loc_and_cov_new = torch.stack([
                                            torch.stack([
                                                torch.stack([
                                                    torch.stack([
                                                        torch.stack( PyDMN.util.conditional( Xnew=Y_time_new, X=self.Y_time,
                                                                                                    kernel=None,
                                                                                                    f_loc=self.gp.coord.loc[v,h,sr_i,lw_i,:], f_scale_tril=self.gp.coord.cov_tril[v,h,sr_i,lw_i,:,:],
                                                                                                    Lff=Lff_coord[lw_i], full_cov=True, whiten=self.whiten, jitter=self.jitter,
                                                                                                    Kfs=Kfs_coord[lw_i], Kss=Kss_coord[lw_i] ) )
                                                    for lw_i in range(self.lw_dim)])
                                                for sr_i in range(self.sr_dim)])
                                            for h in range(self.H_dim)])
                                        for v in range(self.V_net)])

            gp_coord_loc_new = gp_coord_loc_and_cov_new[:,:,:,:,0,0,:]
            gp_coord_cov_new = gp_coord_loc_and_cov_new[:,:,:,:,1,:,:]

            gp_coord_new_sample = torch.stack([
                                    torch.stack([
                                        torch.stack([
                                            torch.stack([
                                    dist.MultivariateNormal( gp_coord_loc_new[v,h,sr_i,lw_i,:] + self.gp.coord.loc_mean[v,h,sr_i,lw_i],
                                                            gp_coord_cov_new[v,h,sr_i,lw_i,:,:] ).expand([num_particles]).sample().transpose(0,1)
                                            for lw_i in range(self.lw_dim)])
                                        for sr_i in range(self.sr_dim)])
                                    for h in range(self.H_dim)])
                                for v in range(self.V_net)])

        ## Generate sociability and popularity for new times ##
        if self.socpop:

            Lff_socpop = []; Kfs_socpop = []; Kss_socpop=[]
            for lw_i in range(self.lw_dim):
                Kff = eval('self.kernel.socpop.'+['link','weight'][lw_i])(self.Y_time).contiguous()
                Kff.view(-1)[::self.T_net + 1] += self.jitter  # add jitter to the diagonal
                Lff_socpop.append( Kff.cholesky() )

                Kfs_socpop.append( eval('self.kernel.socpop.'+['link','weight'][lw_i])(self.Y_time, Y_time_new) )

                Kss_socpop.append( eval('self.kernel.socpop.'+['link','weight'][lw_i])(Y_time_new).contiguous() )
                Kss_socpop[lw_i].view(-1)[::Kss_socpop[lw_i].shape[0] + 1] += self.jitter  # add jitter to the diagonal

            gp_socpop_loc_and_cov_new = torch.stack([
                                            torch.stack([
                                                torch.stack([
                                                    torch.stack( PyDMN.util.conditional( Xnew=Y_time_new, X=self.Y_time,
                                                                                                kernel=None,
                                                                                                f_loc=self.gp.socpop.loc[v,sp_i,lw_i,:], f_scale_tril=self.gp.socpop.cov_tril[v,sp_i,lw_i,:,:],
                                                                                                Lff=Lff_socpop[lw_i], full_cov=True, whiten=self.whiten, jitter=self.jitter,
                                                                                                Kfs=Kfs_socpop[lw_i], Kss=Kss_socpop[lw_i] ) )
                                                for lw_i in range(self.lw_dim)])
                                            for sp_i in range(2)])
                                        for v in range(self.V_net)])

            gp_socpop_loc_new = gp_socpop_loc_and_cov_new[:,:,:,0,0,:]
            gp_socpop_cov_new = gp_socpop_loc_and_cov_new[:,:,:,1,:,:]

            gp_socpop_new_sample = torch.stack([
                                    torch.stack([
                                        torch.stack([
                                dist.MultivariateNormal( gp_socpop_loc_new[v,sp_i,lw_i,:] + self.gp.socpop.loc_mean[v,sp_i,lw_i],
                                                        gp_socpop_cov_new[v,sp_i,lw_i,:,:] ).expand([num_particles]).sample().transpose(0,1)
                                        for lw_i in range(self.lw_dim)])
                                    for sp_i in range(2)])
                            for v in range(self.V_net)])

        ### Calculate Linear Predictors ###
        Y_linpred_new_sample = torch.zeros( self.V_net, self.V_net, Y_time_new.shape[0], self.lw_dim, num_particles )

        if self.coord:
            # gp_coord_new_sample.shape # gp_coord_new_sample[v,h,sr_i,lw_i,t,num_particles]
            send = gp_coord_new_sample[:,:,0,:,:,:].expand( self.V_net, self.V_net, self.H_dim, self.lw_dim, Y_time_new.shape[0], num_particles)
            receive = gp_coord_new_sample[:,:,1,:,:,:].expand( self.V_net, self.V_net, self.H_dim, self.lw_dim, Y_time_new.shape[0], num_particles).transpose(0,1)
            Y_linpred_new_sample += ((send - receive)**2).sum(dim=2).rsqrt().transpose(2,3)

        if self.socpop:
            soc = gp_socpop_new_sample[:,0,:,:,:].expand( self.V_net, self.V_net, self.lw_dim, Y_time_new.shape[0], num_particles)
            pop = gp_socpop_new_sample[:,1,:,:,:].expand( self.V_net, self.V_net, self.lw_dim, Y_time_new.shape[0], num_particles).transpose(0,1)
            Y_linpred_new_sample += (soc + pop).transpose(2,3)

        # Y_linpred_new_sample.mean(dim=Y_linpred_new_sample.dim()-1)
        return Y_linpred_new_sample

    def set_data(self, Y, Y_time, H_dim=3, X=None):
        """
        Sets data for dmn models.
        """

        assert (len(Y.shape)>=3) and (len(Y.shape)<=4)

        # square adjacence matrices
        assert Y.shape[0]==Y.shape[1]

        self.Y = Y
        self.V_net, self.T_net = self.Y.shape[0], self.Y.shape[2]

        self.Y_link = torch.where( Y!=0, torch.ones_like(Y), torch.zeros_like(Y))

        self.K_net = Y.shape[3] if len(Y.shape)==4 else 1

        assert self.T_net == Y_time.shape[0]
        self.Y_time = Y_time

        assert int(H_dim)>=1
        self.H_dim = int(H_dim)

        self.X = X
