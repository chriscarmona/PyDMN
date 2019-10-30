import torch
import matplotlib.pyplot as plt
import pyro.contrib.gp as gp
import numpy as np

def plot_GP_reg( X, y, plot_observed_data=False,
        plot_predictions=False,
        n_prior_samples=0,
        model=None, kernel=None, n_test=100):
    '''
        This helper function does three different things:
        (i) plots the observed data;
        (ii) plots the predictions from the learned GP after conditioning on data;
        (iii) plots samples from the GP prior (with no conditioning on observed data)
    '''
    x_range = [X.min().item(),X.max().item()]

    VI_infer = (True if ((type(model) == gp.models.VariationalSparseGP) or (type(model) == gp.models.VariationalGP))
             else False)

    if plot_predictions:
        Xtest = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if VI_infer:
                mean, cov = model(Xtest, full_cov=True)
                var = cov.diag() + model.likelihood.variance  # standard deviation at each input point x
                sd = var.sqrt()  # standard deviation at each input point x
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
                sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                        (mean - 1.0 * sd).numpy(),
                        (mean + 1.0 * sd).numpy(),
                        color='C0', alpha=0.5)
    if plot_observed_data:
        plt.scatter(model.X.numpy(), model.y.numpy(),marker='o')
        plt.scatter(X.numpy(),y.numpy(),marker='o', facecolors='none', edgecolors='r')
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_test)  # test inputs
        noise = (model.noise if VI_infer
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

def plot_GP_reg_exp( X, y, plot_observed_data=False,
        plot_predictions=False,
        n_prior_samples=0,
        model=None, kernel=None, n_test=100):
    '''
        This helper function does three different things:
        (i) plots the observed data;
        (ii) plots the predictions from the learned GP after conditioning on data;
        (iii) plots samples from the GP prior (with no conditioning on observed data)
    '''
    x_range = [X.min().item(),X.max().item()]

    VI_infer = (True if ((type(model) == gp.models.VariationalSparseGP) or (type(model) == gp.models.VariationalGP))
             else False)

    if plot_predictions:
        Xtest = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if VI_infer:
                mean, cov = model(Xtest, full_cov=True)
                var = cov.diag() + model.likelihood.variance  # standard deviation at each input point x
                sd = var.sqrt()  # standard deviation at each input point x
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
                sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), np.exp( mean.numpy() ), 'r', lw=2)  # plot the mean
        # torch.distributions.normal.Normal(torch.tensor(0.),torch.tensor(1.)).icdf(torch.tensor(0.995))
        # torch.distributions.normal.Normal(torch.tensor(0.),torch.tensor(1.)).cdf(torch.tensor(2))

        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.exp( (mean - 2. * sd).numpy() ),
                         np.exp( (mean + 2. * sd).numpy() ),
                         color='C0', alpha=0.3)
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                        np.exp( (mean - 1. * sd).numpy() ),
                        np.exp( (mean + 1. * sd).numpy() ),
                        color='C0', alpha=0.5)
    if plot_observed_data:
        plt.scatter(model.X.numpy(), np.exp( model.y.numpy() ),marker='o')
        plt.scatter(X.numpy(), np.exp( y.numpy() ),marker='o', facecolors='none', edgecolors='r')
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_test)  # test inputs
        noise = (model.noise if VI_infer
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)


def plot_GP_clas( X, y, plot_observed_data=False,
        plot_predictions=False,
        n_prior_samples=0,
        model=None, kernel=None, n_test=100):
    '''
        This helper function does three different things:
        (i) plots the observed data;
        (ii) plots the predictions from the learned GP after conditioning on data;
        (iii) plots samples from the GP prior (with no conditioning on observed data)
    '''
    x_range = [X.min().item(),X.max().item()]

    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'kx')
    if plot_predictions:
        Xtest = torch.linspace( x_range[0]-3, x_range[1]+3, n_test)  # test inputs
        # compute predictive f_loc and variance
        with torch.no_grad():
            f_loc, f_var = model(Xtest, full_cov=True )
        sd = f_var.diag().sqrt()  # standard deviation at each input point x
        # pred_GP = dist.MultivariateNormal( f_loc, f_var )

        # Get upper and lower confidence bounds
        pred_mean = torch.sigmoid( f_loc )
        plt.plot(Xtest.numpy(), pred_mean, 'r', lw=2)  # plot the f_loc
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the f_loc
                         torch.sigmoid( f_loc - 2.*sd ).numpy(),
                         torch.sigmoid( f_loc + 2.*sd ).numpy(),
                         color='C0', alpha=0.3)
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the f_loc
                         torch.sigmoid( f_loc - 1.*sd ).numpy(),
                         torch.sigmoid( f_loc + 1.*sd ).numpy(),
                         color='C0', alpha=0.5)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_test)  # test inputs
        noise = model.likelihood.variance
        f_var = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=f_var)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.xlim( x_range[0]-1, x_range[1]+1 )
