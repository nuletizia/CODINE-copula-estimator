import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch
import torch.optim as optim
from torch import nn
import scipy.interpolate as interpolate
from math import *
from scipy.stats import gaussian_kde
from scipy.stats import norm, multivariate_normal
from scipy.stats import spearmanr
import scipy.stats as stats
from scipy.optimize import minimize
from copulas.multivariate import gaussian, vine
import copulae


def fit_and_compute_copulas(x_samples, n_samples=1000):
    """
    Fit Gaussian and Vine copulas to the uniform samples and compute their densities
    """
    x_df = pd.DataFrame(x_samples, columns=['U1', 'U2'])
    gaussian_cop = gaussian.GaussianMultivariate()
    gaussian_cop.fit(x_df)
    gaussian_samples = gaussian_cop.sample(n_samples)
    gaussian_samples = gaussian_samples.values
    gaussian_density_values = gaussian_cop.probability_density(x_df)

    # Fit and sample from Vine copula
    vine_cop = vine.VineCopula('center')  # or 'direct' or 'regular'
    vine_cop.fit(x_df)
    vine_samples = vine_cop.sample(n_samples)
    vine_samples = vine_samples.values

    return gaussian_samples, vine_samples, gaussian_density_values


def inverse_transform_sampling(hist, bin_edges):
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist)
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    return inv_cdf


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


def reciprocal_loss(y_true, y_pred):
    return torch.mean(torch.pow(y_true * y_pred, -1))


def my_binary_crossentropy(y_true, y_pred):
    return -torch.mean(torch.log(y_true) + torch.log(y_pred))


def logsumexp_loss(y_true, y_pred):
    loss = torch.logsumexp(y_pred, dim=0) - torch.log(torch.tensor(y_pred.size(0), dtype=torch.float32))
    return loss


def phi(x, mu, sigma):
    N, D = np.shape(x)
    unif_output = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            unif_output[i, j] = (1 + erf((x[i, j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


def generate_mog_samples(batch_size, n_components, latent_dim):
    # Means and covariances for each component
    means = [[0, 0], [-2, -2], [1, -3]]
    covs = [1/10*np.eye(latent_dim), np.eye(latent_dim), 1/4*np.eye(latent_dim)]
    # Weights for the mixture
    weights = [1/3, 1/3, 1/3]

    # Generate samples
    component_indices = np.random.choice(n_components, size=batch_size, p=weights)
    samples = np.zeros((batch_size, latent_dim))
    for i in range(batch_size):
        comp_idx = component_indices[i]
        samples[i] = np.random.multivariate_normal(means[comp_idx], covs[comp_idx])

    return samples


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(100, output_dim)
        )

    def forward(self, input):
        return self.main(input)


class CombinedNet(nn.Module):
    def __init__(self, single_architecture, divergence):
        super(CombinedNet, self).__init__()
        self.div_to_act_func = {
            "GAN": nn.Sigmoid(),
            "KL": nn.Softplus(),
            "HD": nn.Softplus()
        }
        self.divergence = divergence
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[divergence]

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)

        return output_tensor_1, output_tensor_2


def probability_integral_transform(samples):
    n_samples = samples.shape[0]
    n_dims = samples.shape[1]
    u_samples = np.zeros_like(samples)
    for d in range(n_dims):
        # Sort the samples for this dimension
        sorted_samples = np.sort(samples[:, d])
        # Get ranks (empirical CDF values)
        ranks = np.searchsorted(sorted_samples, samples[:, d])
        # Convert to [0,1] scale
        u_samples[:, d] = (ranks + 1) / (n_samples + 1)
    return u_samples


class CODINE():
    def __init__(self, latent_dim, n_components, divergence='KL'):
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.divergence = divergence
        simple_net = Net(latent_dim, 1)
        self.discriminator = CombinedNet(simple_net, divergence)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.002)

    def train(self, epochs, batch_size=40, random_seed=0):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.discriminator.train()
        valid = torch.ones((batch_size, 1))
        fake = torch.zeros((batch_size, 1))

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            # Generate MoG samples
            mog_samples = generate_mog_samples(batch_size, self.n_components, self.latent_dim)

            # Get pseudo-observations via transform sampling
            data_v = torch.Tensor(probability_integral_transform(mog_samples))
            data_v_random = torch.Tensor(np.random.uniform(0, 1, (batch_size, self.latent_dim)))
            D_value_1, D_value_2 = self.discriminator(data_v, data_v_random)

            if self.divergence == 'KL':
                loss_1 = my_binary_crossentropy(valid, D_value_1)
                loss_2 = wasserstein_loss(valid, D_value_2)
                loss = loss_1 + loss_2
                R = D_value_1
                SC = D_value_2

            elif self.divergence == 'GAN':
                loss_fn = torch.nn.BCELoss()
                loss_1 = loss_fn(D_value_1, fake)
                loss_2 = loss_fn(D_value_2, valid)
                loss = loss_1 + loss_2
                R = (1 - D_value_1) / D_value_1
                SC = (1 - D_value_2) / D_value_2

            elif self.divergence == 'HD':
                loss_1 = wasserstein_loss(valid, D_value_1)
                loss_2 = reciprocal_loss(valid, D_value_2)
                loss = loss_1 + loss_2
                R = 1 / (D_value_1 ** 2)
                SC = 1 / (D_value_2 ** 2)

            loss.backward()
            self.optimizer.step()

            copula_estimate = R
            self_consistency = SC

            if epoch % 100 == 0:
                print("%d [D total loss: %f, Copula estimates: %f, Self-consistency mean test: %f]" %
                      (epoch, loss, torch.mean(copula_estimate), torch.mean(self_consistency)))

    def test(self, test_batch=100, test_size=1000):
        copula_density_testbatch = np.zeros((test_batch, test_size))
        c_test_testbatch = np.zeros((test_batch, test_size))
        data_u_testbatch = np.zeros((test_batch, test_size, self.latent_dim))
        data_v_testbatch = np.zeros((test_batch, test_size, self.latent_dim))
        channel_output_testbatch = np.zeros((test_batch, test_size, self.latent_dim))

        for M in range(test_batch):
            mog_samples = generate_mog_samples(test_size, self.n_components, self.latent_dim)
            data_v = torch.Tensor(probability_integral_transform(mog_samples))
            data_v_random = torch.Tensor(np.random.uniform(0, 1, (test_size, self.latent_dim)))
            D_value_1, D_value_2 = self.discriminator(data_v, data_v_random)

            if self.divergence == 'KL':
                copula_estimate = D_value_1.detach().numpy()
                SC = D_value_2.detach().numpy()
            elif self.divergence == 'GAN':
                copula_estimate = ((1 - D_value_1) / D_value_1).detach().numpy()
                SC = ((1 - D_value_2) / D_value_2).detach().numpy()
            elif self.divergence == 'HD':
                copula_estimate = (1 / ((D_value_1) ** 2)).detach().numpy()
                SC = (1 / ((D_value_2) ** 2)).detach().numpy()

            copula_density_testbatch[M, :] = np.squeeze(copula_estimate, axis=1)
            c_test_testbatch[M, :] = np.squeeze(SC, axis=1)
            data_u_testbatch[M, :, :] = probability_integral_transform(mog_samples)
            data_v_testbatch[M, :, :] = data_v
            channel_output_testbatch[M, :, :] = mog_samples

        return copula_density_testbatch, c_test_testbatch, data_u_testbatch, data_v_testbatch, channel_output_testbatch


def compute_true_copula_kde(samples, n_grid=100, bandwidth='scott'):

    n_samples = samples.shape[0]

    # Compute empirical CDFs
    sorted_x = np.sort(samples[:, 0])
    sorted_y = np.sort(samples[:, 1])

    # Get ranks for each dimension
    ranks_x = np.searchsorted(sorted_x, samples[:, 0]) / n_samples
    ranks_y = np.searchsorted(sorted_y, samples[:, 1]) / n_samples

    # Stack coordinates for KDE
    copula_samples = np.vstack([ranks_x, ranks_y])

    # Fit KDE
    kde = gaussian_kde(copula_samples, bw_method=bandwidth)

    # Create grid for density evaluation
    u_grid = np.linspace(0, 1, n_grid)
    v_grid = np.linspace(0, 1, n_grid)
    U, V = np.meshgrid(u_grid, v_grid)
    positions = np.vstack([U.ravel(), V.ravel()])

    # Evaluate KDE on grid
    true_copula = kde(positions).reshape(n_grid, n_grid)
    return true_copula, ranks_x, ranks_y, U, V


def save_comparison_results_total(copula_density_codine, data_v, channel_output, n_components, divergence, test_size=1000,
                            n_grid=100, bandwidth=0.05):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Compute true copula with KDE
    true_copula, ranks_x, ranks_y, U, V = compute_true_copula_kde(
        channel_output[0],
        n_grid=n_grid,
        bandwidth=bandwidth
    )

    fig = plt.figure(figsize=(30, 10))

    # 1. Original MoG 3D Scatter
    ax1 = fig.add_subplot(131, projection='3d')
    # Compute density at MoG points for coloring
    kde_mog = gaussian_kde(channel_output[0].T)
    density_mog = kde_mog(channel_output[0].T)
    scatter1 = ax1.scatter(channel_output[0, :, 0],
                           channel_output[0, :, 1],
                           density_mog,
                           c=density_mog,
                           cmap='plasma')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Density')
    plt.colorbar(scatter1, ax=ax1)

    # 2. Uniform Samples 3D Scatter (after PIT)
    ax3 = fig.add_subplot(132, projection='3d')
    kde_uniform = gaussian_kde(np.vstack([ranks_x, ranks_y]))
    density_uniform = kde_uniform(np.vstack([ranks_x, ranks_y]))
    scatter2 = ax3.scatter(ranks_x,
                           ranks_y,
                           density_uniform,
                           c=density_uniform,
                           cmap='viridis')
    ax3.set_xlabel('U1')
    ax3.set_ylabel('U2')
    ax3.set_zlabel('Density')
    plt.colorbar(scatter2, ax=ax3)

    # 3. Estimated Copula Density (3D)
    ax4 = fig.add_subplot(133, projection='3d')
    scatter3 = ax4.scatter(data_v[0, :, 0],
                           data_v[0, :, 1],
                           copula_density_codine[0, :],
                           c=copula_density_codine[0, :],
                           cmap='viridis')
    ax4.set_xlabel('U1')
    ax4.set_ylabel('U2')
    ax4.set_zlabel('Density')
    plt.colorbar(scatter3, ax=ax4)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'plots/total_copula_comparison_{divergence}_{n_components}_components.png',
                bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5000) # 5000
    parser.add_argument('--test_size', type=int, default=20000) # 10000
    parser.add_argument('--test_batch', type=int, default=10)
    parser.add_argument('--divergence', type=str, default='KL')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    # Initialize CODINE with MoG
    codine = CODINE(args.latent_dim, args.n_components, args.divergence)

    # Train
    codine.train(epochs=args.epochs, batch_size=args.batch_size)

    # Test
    copula_density_codine, c_test, data_u, data_v, channel_output = codine.test(
        test_batch=args.test_batch,
        test_size=args.test_size
    )

    save_comparison_results_total(
        copula_density_codine,
        data_v,
        channel_output,
        args.n_components,
        args.divergence,
        args.test_size,
        n_grid=100,  # Increase for smoother visualization
        bandwidth=0.05  # Adjust this value to control smoothness
    )


