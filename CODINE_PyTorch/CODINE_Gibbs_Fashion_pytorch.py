from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from math import *
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os
from torchvision import datasets


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
    return torch.logsumexp(y_pred, dim=0) - torch.log(torch.tensor(y_pred.size(0), dtype=torch.float32))


def phi(x, mu, sigma):
    N, D = x.shape
    unif_output = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 16, 2, padding=1),
            nn.Conv2d(16, 4, 2, padding=1),
            nn.ReLU(),
        )
        self.feature_size = self._get_conv_output_size()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.feature_size, latent_dim)
        self.activation = nn.Sigmoid()

    def _get_conv_output_size(self):
        x = torch.randn(1, 1, 28, 28)
        x = self.conv_layers(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 49)
        self.activation = nn.ReLU()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = x.view(-1, 1, 7, 7)
        x = self.conv_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim, divergence):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Linear(128, 50),
            nn.Sigmoid(),
            nn.Linear(50, 1)
        )
        if divergence == 'GAN':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softplus()

    def forward(self, x):
        x = self.model(x)
        x = self.final_activation(x)
        return x


class CODINE():
    def __init__(self, latent_dim, divergence='KL', alpha=1, watch_training=False):
        self.latent_dim = latent_dim
        self.divergence = divergence
        self.alpha = alpha
        self.watch_training = watch_training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: ", self.device)

        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)
        self.discriminator = Discriminator(latent_dim, divergence).to(self.device)

        self.ae_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.002)

    def train_autoencoder(self, train_data, epochs, batch_size=40):
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.FloatTensor(train_data).to(self.device)
        
        criterion = nn.MSELoss()
        N = train_data.size(0)
        
        for epoch in range(epochs):
            idx = np.random.randint(0, N, batch_size)
            data_batch = train_data[idx].reshape(-1, 1, 28, 28).to(self.device)
            # Train autoencoder
            self.ae_optimizer.zero_grad()
            encoded = self.encoder(data_batch)
            decoded = self.decoder(encoded)
            loss = criterion(decoded, data_batch)
            loss.backward()
            self.ae_optimizer.step()
            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] [AE loss: {loss.item():.4f}]")

    def train(self, train_data, epochs, batch_size=40):
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.FloatTensor(train_data).to(self.device)
        with torch.no_grad():
            train_data_reshaped = torch.reshape(train_data, (60000, 1, 28, 28))
            data_z_full = self.encoder(train_data_reshaped)
        N = train_data.size(0)
        data_sorted, _ = torch.sort(data_z_full, dim=0)
        cdf_x = torch.zeros((N, self.latent_dim))
        for i in range(self.latent_dim):
            cdf_x[:,i] = torch.tensor(1. * np.arange(len(data_z_full[:,i])) / (len(data_z_full[:,i]) - 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, N, batch_size)
            # Get latent representation
            with torch.no_grad():
                data_z = data_z_full[idx]
            # Transform to uniform via empirical CDF
            data_u = torch.zeros_like(data_z)
            for i in range(self.latent_dim):
                sorted_values = data_sorted[:, i].contiguous().to(self.device)
                z_values = data_z[:, i].contiguous().to(self.device)
                ranks = torch.searchsorted(sorted_values, z_values)
                ranks = torch.clamp(ranks, 0, N-1)  # Ensure indices are within bounds
                data_u[:, i] = cdf_x[ranks, i]

            # Generate uniform random samples
            data_pi = torch.rand(batch_size, self.latent_dim).to(self.device)
            valid = torch.ones(batch_size, 1).to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()
            d_u = self.discriminator(data_u)
            d_pi = self.discriminator(data_pi)

            if self.divergence == 'KL':
                d_loss = my_binary_crossentropy(valid, d_u) + wasserstein_loss(valid, d_pi)
            elif self.divergence == 'GAN':
                d_loss = nn.BCELoss()(d_u, torch.zeros_like(d_u)) + nn.BCELoss()(d_pi, valid)
            elif self.divergence == 'HD':
                d_loss = wasserstein_loss(valid, d_u) + reciprocal_loss(valid, d_pi)

            d_loss.backward()
            self.d_optimizer.step()

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

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] [D loss: {d_loss.item():.4f}] "
                    f"[Copula est: {torch.mean(R).item():.4f}] "
                    f"[Self-consistency: {torch.mean(SC).item():.4f}]")

    def test(self, test_data, test_batch=100):
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.FloatTensor(test_data).to(self.device)
            
        N = test_data.size(0)
        test_size = N // test_batch
        
        copula_density_testbatch = torch.zeros(test_batch, test_size)
        data_u_testbatch = torch.zeros(test_batch, test_size, self.latent_dim)
        data_pi_testbatch = torch.zeros(test_batch, test_size, self.latent_dim)

        data_sorted = torch.sort(test_data, dim=0)[0]
        cdf_x = torch.linspace(0, 1, N, device=self.device)

        with torch.no_grad():
            for M in range(test_batch):
                idx = np.random.randint(0, N, test_size)
                data_batch = test_data[idx].reshape(-1, 1, 28, 28).to(self.device)
                data_z = self.encoder(data_batch)

                data_u = torch.zeros_like(data_z)
                for i in range(self.latent_dim):
                    sorted_values = data_sorted[:, i].contiguous().to(self.device)
                    z_values = data_z[:, i].contiguous().to(self.device)
                    ranks = torch.searchsorted(sorted_values, z_values)
                    ranks = torch.clamp(ranks, 0, N - 1)
                    data_u[:, i] = cdf_x[ranks]

                data_pi = torch.rand(test_size, self.latent_dim).to(self.device)
                d_u = self.discriminator(data_u)

                if self.divergence == 'KL':
                    copula_estimate = d_u
                elif self.divergence == 'GAN':
                    copula_estimate = (1 - d_u) / d_u
                elif self.divergence == 'HD':
                    copula_estimate = 1 / (d_u**2)

                copula_density_testbatch[M] = copula_estimate.squeeze()
                data_u_testbatch[M] = data_u
                data_pi_testbatch[M] = data_pi

        return (copula_density_testbatch.cpu().numpy(), 
                data_u_testbatch.cpu().numpy(), 
                data_pi_testbatch.cpu().numpy())

    def copula_gibbs_sampling(self, grid_points=10, test_size=1000):
        uv_samples = torch.zeros(test_size, self.latent_dim)
        uv_samples[0] = torch.rand(self.latent_dim)
        a = torch.linspace(0, 1, grid_points, device=self.device)

        with torch.no_grad():
            for t in range(1, test_size):
                for i in range(self.latent_dim):
                    if i == 0:
                        uv_i_vector = torch.cat([
                            a.view(-1, 1),
                            uv_samples[t-1, i+1:].repeat(grid_points, 1)
                        ], dim=1)
                    elif i < self.latent_dim - 1:
                        uv_i_vector = torch.cat([
                            uv_samples[t, :i].repeat(grid_points, 1),
                            a.view(-1, 1),
                            uv_samples[t-1, i+1:].repeat(grid_points, 1)
                        ], dim=1)
                    else:
                        uv_i_vector = torch.cat([
                            uv_samples[t, :i].repeat(grid_points, 1),
                            a.view(-1, 1)
                        ], dim=1)
                    disc_output = self.discriminator(uv_i_vector.to(self.device))

                    if self.divergence == 'KL':
                        copula_density_vector = disc_output
                    elif self.divergence == 'GAN':
                        copula_density_vector = (1 - disc_output) / disc_output
                    elif self.divergence == 'HD':
                        copula_density_vector = 1 / (disc_output**2)

                    copula_density_vector = copula_density_vector.cpu().numpy()
                    copula_density_vector = copula_density_vector/np.sum(copula_density_vector)
                    icdf = inverse_transform_sampling(
                        np.squeeze(copula_density_vector),
                        np.linspace(0, 1, grid_points+1)
                    )
                    uv_samples[t, i] = torch.tensor(float(icdf(np.random.uniform(0, 1))))

        return uv_samples.numpy()

    def data_sampling(self, uv_samples, train_data, grid_points=10):
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.FloatTensor(train_data).to(self.device)
        if not isinstance(uv_samples, torch.Tensor):
            uv_samples = torch.FloatTensor(uv_samples).to(self.device)
        train_data_reshaped = train_data.reshape(-1, 1, 28, 28)
        with torch.no_grad():
            train_z = self.encoder(train_data_reshaped)

        xy_samples = torch.zeros_like(uv_samples)
        for i in range(self.latent_dim):
            hist, bin_edges = np.histogram(train_z[:, i].cpu().numpy(), bins=grid_points, density=True)
            hist = hist / np.sum(hist)
            icdf = inverse_transform_sampling(hist, bin_edges)

            for t in range(uv_samples.size(0)):
                xy_samples[t, i] = torch.tensor(float(icdf(uv_samples[t, i].item())))

        xy_samples = xy_samples.to(self.device)
        generated_images = self.decoder(xy_samples)
        return generated_images.detach().numpy()


def save_generated_images(xy_generated_full, uv_generated, name, save_dir="Images", only_grid=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    xy_generated = xy_generated_full[5000:5100]
    xy_generated[50:100] = xy_generated_full[9000:9050]
    if only_grid:
        # Save grid of generated images
        plt.figure(figsize=(20, 20))
        for i in range(min(100, len(xy_generated))):
            plt.subplot(10, 10, i + 1)
            plt.imshow(denormalize(torch.tensor(xy_generated[i].reshape(28, 28))).detach().numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(save_dir, '{}_grid.png'.format(name)))
        plt.close()
    else:
        # Save individual generated images
        for i in range(min(100, len(xy_generated))):
            plt.figure(figsize=(4, 4))
            plt.imshow(denormalize(torch.tensor(xy_generated[i].reshape(28, 28))).detach().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'{name}_{i}.png'))
            plt.close()

        # Save grid of generated images
        plt.figure(figsize=(20, 20))
        for i in range(min(100, len(xy_generated))):
            plt.subplot(10, 10, i + 1)
            plt.imshow(denormalize(torch.tensor(xy_generated[i].reshape(28, 28))).detach().numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(save_dir, '{}_grid.png'.format(name)))
        plt.close()

        # Save copula samples (uv_generated)
        plt.figure(figsize=(10, 10))
        plt.scatter(uv_generated[:, 0], uv_generated[:, 1], alpha=0.5)
        plt.title('Copula Samples')
        plt.xlabel('U')
        plt.ylabel('V')
        plt.savefig(os.path.join(save_dir, '{}_copula_samples.png'.format(name)))
        plt.close()


def denormalize(normalized_image, mean=0.5, std=0.5):
    denormalized = normalized_image * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)*255
    denormalized = 255-denormalized
    return denormalized


def save_fashion_mnist_grid(grid_size=10, save_path='fashion_mnist_grid.png'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=grid_size * grid_size,
        shuffle=True
    )
    images, labels = next(iter(test_loader))
    plt.figure(figsize=(20, 20))
    for i in range(min(100, len(images))):
        plt.subplot(10, 10, i + 1)
        plt.imshow(denormalize(images[i].reshape(28, 28)).detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--epochs_ae', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--test_batch', type=int, default=1)
    parser.add_argument('--divergence', type=str, default='GAN')
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    # Load Fashion MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        fashion_mnist,
        batch_size=len(fashion_mnist),
        shuffle=True
    )
    train_data = next(iter(train_loader))[0].view(-1, 28*28)

    # Initialize CODINE
    codine = CODINE(latent_dim=args.latent_dim, divergence=args.divergence)
    
    # Train autoencoder and CODINE
    codine.train_autoencoder(train_data, epochs=args.epochs_ae, batch_size=args.batch_size)
    codine.train(train_data, epochs=args.epochs, batch_size=args.batch_size)
    
    # Test and generate samples
    copula_density, data_u, data_u_random = codine.test(
        train_data,
        test_batch=args.test_batch
    )
    
    uv_generated = codine.copula_gibbs_sampling(grid_points=30, test_size=10000)
    xy_generated = codine.data_sampling(uv_generated, train_data, grid_points=30)
    save_generated_images(xy_generated, uv_generated, name="generated_images")

