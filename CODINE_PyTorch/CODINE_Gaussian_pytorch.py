import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
from torch import nn
import scipy.interpolate as interpolate
from math import *


def inverse_transform_sampling(hist, bin_edges):
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist)
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    return inv_cdf


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true* y_pred)


def reciprocal_loss(y_true, y_pred):
    return torch.mean(torch.pow(y_true*y_pred,-1))


def my_binary_crossentropy(y_true, y_pred):
    return -torch.mean(torch.log(y_true)+torch.log(y_pred))


def logsumexp_loss(y_true, y_pred):
    loss = torch.logsumexp(y_pred) - torch.log(y_true.shape[0].float)
    return loss


def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


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
            "HD": nn.Softplus(),
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


class CODINE():
    def __init__(self, latent_dim, EbN0, divergence='KL', rho=0):
        self.latent_dim = latent_dim
        self.EbN0 = EbN0
        self.divergence = divergence  # type of f-divergence to use for training and estimation
        eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))
        self.eps = eps
        self.rho = rho
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

        rho = self.rho
        Sigma_N = np.zeros((self.latent_dim, self.latent_dim))

        # Define the cov matrix
        for i in range(self.latent_dim - 1):
            Sigma_N[i, i + 1] = 1
            Sigma_N[i + 1, i] = 1
        Sigma_N = np.eye(self.latent_dim) + rho * Sigma_N

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            channel_input = np.random.normal(0, 1, (batch_size, self.latent_dim))
            eps = self.eps
            channel_output = channel_input + eps * np.random.multivariate_normal(np.zeros((self.latent_dim,)),Sigma_N,batch_size)

            data_v = torch.Tensor(phi(channel_output, 0, np.sqrt(1+eps**2)))
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
                R = 1 / (D_value_1**2)
                SC = 1 / (D_value_2**2)

            loss.backward()
            self.optimizer.step()

            copula_estimate = R
            self_consistency = SC
            # Plot the progress
            print("%d [D total loss : %f, Copula estimates: %f, Self-consistency mean test: %f]" % (epoch, loss, torch.mean(copula_estimate), torch.mean(self_consistency)))


    def test(self, test_batch=100, test_size=1000):
        eps = self.eps
        rho = self.rho

        copula_density_testbatch = np.zeros((test_batch, test_size))
        c_test_testbatch = np.zeros((test_batch, test_size))

        data_u_testbatch = np.zeros((test_batch, test_size,self.latent_dim))
        data_v_testbatch = np.zeros((test_batch, test_size,self.latent_dim))

        Sigma_N = np.zeros((self.latent_dim,self.latent_dim))
        for i in range(self.latent_dim-1):
            Sigma_N[i, i + 1] = 1
            Sigma_N[i + 1, i] = 1
        Sigma_N = np.eye(self.latent_dim)+rho*Sigma_N

        for M in range(test_batch):
            channel_input = np.random.normal(0, 1, (test_size, self.latent_dim))
            data_u = phi(channel_input,0,1)

            channel_output = channel_input + eps * np.random.multivariate_normal(np.zeros((self.latent_dim,)),Sigma_N,test_size)
            data_v = torch.Tensor(phi(channel_output,0,np.sqrt(1+eps**2)))
            data_v_random = torch.Tensor(np.random.uniform(0,1,(test_size, self.latent_dim)))

            D_value_1, D_value_2 = self.discriminator(data_v, data_v_random)

            if self.divergence == 'KL':
                copula_estimate = D_value_1.detach().numpy()
                SC = D_value_2.detach().numpy()
            elif self.divergence == 'GAN':
                copula_estimate = ((1 - D_value_1) / D_value_1).detach().numpy()
                SC = ((1 - D_value_2) / D_value_2).detach().numpy()
            elif self.divergence == 'HD':
                copula_estimate = (1 / ((D_value_1)**2)).detach().numpy()
                SC = (1 / ((D_value_2)**2)).detach().numpy()

            copula_density_testbatch[M, :] = np.squeeze(copula_estimate, axis=1)
            c_test_testbatch[M, :] = np.squeeze(SC, axis=1)
            data_u_testbatch[M,:,:] = data_u
            data_v_testbatch[M,:,:] = data_v

        return copula_density_testbatch, c_test_testbatch, data_u_testbatch, data_v_testbatch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=256)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=5000)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)
    parser.add_argument('--test_batch', help='Number average estimators', default=10)
    parser.add_argument('--divergence', help='f-divergence measure', default='KL')
    parser.add_argument('--rho', help='off-diagonal noise correlation coefficient', default=0.5)
    parser.add_argument('--latent_dim', help='d-dimension', default=2)
    parser.add_argument('--save', help='save results for matlab', default=False)
    args = parser.parse_args()

    test_size = int(args.test_size)
    test_batch = int(args.test_batch)
    divergence = str(args.divergence)
    rho = float(args.rho)
    latent_dim = int(args.latent_dim)
    save_on_matlab = bool(args.save)

    SNR_dB = range(0, 1)

    copula_density_total = np.zeros((len(SNR_dB),test_batch,test_size))
    c_test_total = np.zeros((len(SNR_dB),test_batch,test_size))
    data_u_total = np.zeros((len(SNR_dB),test_batch,test_size,latent_dim))
    data_v_total = np.zeros((len(SNR_dB),test_batch,test_size,latent_dim))

    j = 0
    for SNR in SNR_dB:
        print(f'Actual SNR is:{SNR}')
        # Initialize CODINE
        codine = CODINE(latent_dim, SNR, divergence, rho)
        # Train
        codine.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        copula_density, c_test, data_u, data_v = codine.test(test_batch=test_batch, test_size=test_size)

        copula_density_total[j,:,:] = copula_density
        c_test_total[j,:,:] = c_test
        data_u_total[j,:,:,:] = data_u
        data_v_total[j,:,:,:] = data_v

        # plot the Gaussian copula density estimate
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(data_v[0, :, 0], data_v[0, :, 1], copula_density[0,:], color="green")
        plt.title("Gaussian copula density estimate with CODINE")

        # To show the plot
        plt.show()

        del codine
        j = j+1

