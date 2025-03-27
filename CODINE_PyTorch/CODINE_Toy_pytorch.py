from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import argparse
from math import *
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


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
    return torch.logsumexp(y_pred, dim=0) - torch.log(torch.tensor(y_true.shape[0], dtype=torch.float32))


def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


class Discriminator(nn.Module):
    def __init__(self, latent_dim, divergence):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid() if divergence == 'GAN' else nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)


class CODINE():
    def __init__(self, latent_dim, divergence='KL'):
        self.latent_dim = latent_dim
        self.divergence = divergence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize discriminator
        self.discriminator = Discriminator(latent_dim, divergence).to(self.device)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

    def train(self, epochs, batch_size=40):
        # Initialize arrays
        data_u = np.zeros((batch_size, self.latent_dim))
        data_sorted_x = np.zeros((10000, self.latent_dim))
        cdf_x = np.zeros((10000, self.latent_dim))

        # Generate initial data and estimate CDF
        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

        for i in range(self.latent_dim):
            data_sorted_x[:,i] = np.sort(data[:,i])
            cdf_x[:,i] = 1. * np.arange(len(data[:,i])) / (len(data[:,i]) - 1)

        for epoch in range(epochs):
            # Generate data
            time = np.random.normal(0, 1, (batch_size, 1))
            noise = 0.1 * np.random.normal(0, 1, (batch_size, self.latent_dim))
            data_x = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

            # Transform data
            for i in range(self.latent_dim):
                for j in range(batch_size):
                    data_u[j,i] = cdf_x[np.argmin(np.abs(data_x[j,i]-data_sorted_x[:,i])),i]

            data_pi = np.random.uniform(0, 1, (batch_size, self.latent_dim))

            # Convert to PyTorch tensors
            data_u_tensor = torch.FloatTensor(data_u).to(self.device)
            data_pi_tensor = torch.FloatTensor(data_pi).to(self.device)
            valid = torch.ones(batch_size, 1).to(self.device)

            # Train discriminator
            self.optimizer.zero_grad()
            
            d_u = self.discriminator(data_u_tensor)
            d_pi = self.discriminator(data_pi_tensor)

            if self.divergence == 'KL':
                loss = my_binary_crossentropy(valid, d_u) + wasserstein_loss(valid, d_pi)
            elif self.divergence == 'GAN':
                loss = nn.BCELoss()(d_u, torch.zeros_like(d_u)) + nn.BCELoss()(d_pi, valid)
            elif self.divergence == 'HD':
                loss = wasserstein_loss(valid, d_u) + reciprocal_loss(valid, d_pi)

            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                if self.divergence == 'KL':
                    R = d_u
                    SC = d_pi
                elif self.divergence == 'GAN':
                    R = (1 - d_u) / d_u
                    SC = (1 - d_pi) / d_pi
                elif self.divergence == 'HD':
                    R = 1 / (d_u**2)
                    SC = 1 / (d_pi**2)

            print(f"{epoch} [D loss: {loss.item():.4f}, Copula estimates: {torch.mean(R).item():.4f}, "
                  f"Self-consistency mean test: {torch.mean(SC).item():.4f}]")

    def test(self, test_batch=100, test_size=1000):
        copula_density_testbatch = np.zeros((test_batch, test_size))
        data_u_testbatch = np.zeros((test_batch, test_size, self.latent_dim))
        data_pi_testbatch = np.zeros((test_batch, test_size, self.latent_dim))
        data_u = np.zeros((test_size, self.latent_dim))

        data_sorted_x = np.zeros((10000, self.latent_dim))
        cdf_x = np.zeros((10000, self.latent_dim))

        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

        for i in range(self.latent_dim):
            data_sorted_x[:, i] = np.sort(data[:, i])
            cdf_x[:, i] = 1. * np.arange(len(data[:, i])) / (len(data[:, i]) - 1)

        with torch.no_grad():
            for M in range(test_batch):
                # Generate observations and pseudo-observations
                time = np.random.normal(0, 1, (test_size, 1))
                noise = 0.1 * np.random.normal(0, 1, (test_size, self.latent_dim))
                data_x = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

                # Transform data
                for i in range(self.latent_dim):
                    for j in range(test_size):
                        data_u[j, i] = cdf_x[np.argmin(np.abs(data_x[j, i] - data_sorted_x[:, i])), i]

                data_pi = np.random.uniform(0, 1, (test_size, self.latent_dim))

                data_u_tensor = torch.FloatTensor(data_u).to(self.device)
                
                D_value_1 = self.discriminator(data_u_tensor)
                D_value_1 = D_value_1.cpu().numpy()

                if self.divergence == 'KL':
                    copula_estimate = D_value_1
                elif self.divergence == 'GAN':
                    copula_estimate = (1 - D_value_1) / D_value_1
                elif self.divergence == 'HD':
                    copula_estimate = 1 / (D_value_1**2)

                copula_density_testbatch[M, :] = np.squeeze(copula_estimate, axis=1)
                data_u_testbatch[M, :, :] = data_u
                data_pi_testbatch[M, :, :] = data_pi

        return copula_density_testbatch, data_u_testbatch, data_pi_testbatch

    def copula_gibbs_sampling(self, grid_points=10, test_size=1000):
        # Sample from Copula using Gibbs sampling mechanism
        a = np.linspace(0, 1, grid_points)
        uv_samples = np.zeros((test_size, self.latent_dim))
        uv_samples[0, :] = np.random.uniform(0, 1, self.latent_dim)

        with torch.no_grad():
            for t in range(1, test_size):
                # For every component
                for i in range(self.latent_dim):
                    if i == 0:
                        uv_i_vector = np.concatenate((
                            a.reshape(-1, 1),
                            np.repeat(uv_samples[t-1, i+1:self.latent_dim], repeats=grid_points, axis=0).reshape(-1, 1)
                        ), axis=1)
                    elif i > 0 and i < self.latent_dim-1:
                        uv_i_vector_left = np.concatenate((
                            np.repeat(uv_samples[t, 0:i], repeats=grid_points, axis=0).reshape(-1, 1),
                            a.reshape(-1, 1)
                        ), axis=1)
                        uv_i_vector = np.concatenate((
                            uv_i_vector_left,
                            np.repeat(uv_samples[t-1, i+1:self.latent_dim], repeats=grid_points, axis=0).reshape(-1, 1)
                        ), axis=1)
                    else:
                        uv_i_vector = np.concatenate((
                            np.repeat(uv_samples[t, 0:i], repeats=grid_points, axis=0).reshape(-1, 1),
                            a.reshape(-1, 1)
                        ), axis=1)

                    uv_i_tensor = torch.FloatTensor(uv_i_vector).to(self.device)
                    disc_output = self.discriminator(uv_i_tensor).cpu().numpy()

                    if self.divergence == 'KL':
                        copula_density_vector = disc_output
                    elif self.divergence == 'GAN':
                        copula_density_vector = (1 - disc_output) / disc_output
                    elif self.divergence == 'HD':
                        copula_density_vector = 1 / (disc_output ** 2)

                    copula_density_vector = copula_density_vector/np.sum(copula_density_vector)
                    icdf = inverse_transform_sampling(np.squeeze(copula_density_vector), np.linspace(0, 1, grid_points+1))
                    unif_source = np.random.uniform(0, 1)
                    uv_samples[t, i] = icdf(unif_source)

        return uv_samples

    def data_sampling(self, uv_samples, grid_points=10):
        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

        xy_samples = np.zeros((10000, self.latent_dim))

        for i in range(self.latent_dim):
            hist, bin_edges = np.histogram(data[:, i], bins=grid_points, density=True)
            hist = hist / np.sum(hist)
            icdf = inverse_transform_sampling(hist, bin_edges)

            for t in range(10000):
                xy_samples[t, i] = icdf(uv_samples[t, i])

        plt.scatter(data[:, 0], data[:, 1], c="red")
        plt.scatter(xy_samples[:, 0], xy_samples[:, 1], c="blue")
        plt.show()

        return xy_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=256)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=5000)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)
    parser.add_argument('--test_batch', help='Number average estimators', default=10)
    parser.add_argument('--divergence', help='f-divergence measure', default='GAN')
    parser.add_argument('--latent_dim', help='d-dimension', default=2)
    parser.add_argument('--save', help='save results for matlab', default=False)

    args = parser.parse_args()

    test_size = int(args.test_size)
    test_batch = int(args.test_batch)
    divergence = str(args.divergence)
    latent_dim = int(args.latent_dim)
    save_on_matlab = bool(args.save)

    # Initialize CODINE
    codine = CODINE(latent_dim, divergence)
    # Train
    codine.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
    # Test
    copula_density, data_u, data_u_random = codine.test(test_batch=test_batch, test_size=test_size)
    # Sample via Gibbs, generate pseudo-observation
    uv_generated = codine.copula_gibbs_sampling(grid_points=30, test_size=10000)
    # Inverse transform sampling, get new observations, scatter plot
    xy_generated = codine.data_sampling(uv_generated, grid_points=30)

    # Save on Matlab
    if save_on_matlab:
        sio.savemat('CODINE_Gibbs_Toy.mat', {
            'xy_generated': xy_generated,
            'uv_generated': uv_generated,
            'c_u': copula_density,
            'u': data_u,
            'u_pi': data_u_random
        })