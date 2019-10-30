import torch
import matplotlib.pyplot as plt
import pyro.contrib.gp as gp

def links( model,
        agent_i=0, agent_j=1,
        Y=None, Y_time=None,
        plot_observed_data=False, plot_predictions=False, n_grid=100, num_particles=100):
    '''
        This helper function does three different things:
        (i) plots the observed data;
        (ii) plots the predictions from the learned GP after conditioning on data;
        (iii) plots samples from the GP prior (with no conditioning on observed data)
    '''
    assert agent_i != agent_j

    Y_link = torch.where( Y!=0, torch.ones_like(Y), torch.zeros_like(Y))
    x_range = [Y_time.min().item(),Y_time.max().item()]

    if plot_predictions:
        Y_time_grid = torch.linspace( x_range[0]-0.5, x_range[1]+0.5, n_grid)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            Y_sample_grid = model(Y_time_grid, num_particles)
            Y_mean_hat = Y_sample_grid.mean(dim=Y_sample_grid.dim()-1)
            Y_std_hat = Y_sample_grid.std(dim=Y_sample_grid.dim()-1)

        if model.weighted:
            plt.subplot(2, 1, 1)

        plt.plot(Y_time_grid.numpy(), torch.sigmoid( Y_mean_hat[agent_i,agent_j,:,0] ).numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Y_time_grid.numpy(),  # plot the two-sigma uncertainty about the mean
                         torch.sigmoid( Y_mean_hat[agent_i,agent_j,:,0] - 2*Y_std_hat[agent_i,agent_j,:,0] ).numpy(),
                         torch.sigmoid( Y_mean_hat[agent_i,agent_j,:,0] + 2*Y_std_hat[agent_i,agent_j,:,0] ).numpy(),
                         color='C0', alpha=0.3)
        plt.fill_between(Y_time_grid.numpy(),  # plot the two-sigma uncertainty about the mean
                         torch.sigmoid( Y_mean_hat[agent_i,agent_j,:,0] - 1*Y_std_hat[agent_i,agent_j,:,0] ).numpy(),
                         torch.sigmoid( Y_mean_hat[agent_i,agent_j,:,0] + 1*Y_std_hat[agent_i,agent_j,:,0] ).numpy(),
                         color='C0', alpha=0.5)
        plt.ylim(-0.1, 1.1)

        if model.weighted:
            plt.subplot(2, 1, 2)
            plt.plot(Y_time_grid.numpy(), ( Y_mean_hat[agent_i,agent_j,:,1] ).numpy(), 'r', lw=2)  # plot the mean
            plt.fill_between(Y_time_grid.numpy(),  # plot the two-sigma uncertainty about the mean
                             ( Y_mean_hat[agent_i,agent_j,:,1] - 2*Y_std_hat[agent_i,agent_j,:,1] ).numpy(),
                             ( Y_mean_hat[agent_i,agent_j,:,1] + 2*Y_std_hat[agent_i,agent_j,:,1] ).numpy(),
                             color='C0', alpha=0.3)
            plt.fill_between(Y_time_grid.numpy(),  # plot the two-sigma uncertainty about the mean
                             ( Y_mean_hat[agent_i,agent_j,:,1] - 1*Y_std_hat[agent_i,agent_j,:,1] ).numpy(),
                             ( Y_mean_hat[agent_i,agent_j,:,1] + 1*Y_std_hat[agent_i,agent_j,:,1] ).numpy(),
                             color='C0', alpha=0.5)

    if plot_observed_data:
        if model.weighted:
            plt.subplot(2, 1, 1)
        plt.scatter(model.Y_time.numpy(), model.Y_link[agent_i,agent_j,:].numpy(),marker='o')
        plt.scatter(Y_time.numpy(), Y_link[agent_i,agent_j,:].numpy(),marker='o', facecolors='none', edgecolors='b')
        plt.ylim(-0.1, 1.1)

        if model.weighted:
            plt.subplot(2, 1, 2)
            plt.scatter(model.Y_time.numpy(), model.Y[agent_i,agent_j,:].numpy(),marker='o')
            plt.scatter(Y_time.numpy(), Y[agent_i,agent_j,:].numpy(),marker='o', facecolors='none', edgecolors='b')

    plt.show()
