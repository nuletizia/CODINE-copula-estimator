from __future__ import print_function, division

from keras.layers import Input, Dense, GaussianNoise, LeakyReLU,BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
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
    return K.mean(y_true* y_pred)

def reciprocal_loss(y_true, y_pred):
    return K.mean(K.pow(y_true*y_pred,-1))

def my_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.log(y_true)+K.log(y_pred))

def logsumexp_loss(y_true, y_pred):
    loss = K.logsumexp(y_pred) - K.log(tf.cast(K.shape(y_true)[0], tf.float32))
    return loss

def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output

class CODINE():
    def __init__(self, latent_dim, divergence = 'KL'):

        # Input shape
        self.latent_dim = latent_dim
        self.divergence = divergence # type of f-divergence to use for training and estimation
        self.last_layer_activation = 'softplus'

        # Noise std based on EbN0 in dB

        optimizer = Adam(0.002, 0.5)

        # Build and compile the discriminator (CODINE has a discriminative formulation)
        self.discriminator = self.build_discriminator()

        u = Input(shape=(self.latent_dim,))
        pi = Input(shape=(self.latent_dim,))

        # The discriminator takes as input joint or marginal vectors
        d_u = self.discriminator(u)
        d_pi = self.discriminator(pi)

        # Train the discriminator
        self.combined = Model([u, pi], [d_u,d_pi])

        # choose the loss function based on the f-divergence type
        if self.divergence == 'KL':
            self.combined.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[1,1], optimizer=optimizer)
        elif self.divergence == 'GAN':
            self.combined.compile(loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[1,1], optimizer=optimizer)
        elif self.divergence == 'HD':
            self.combined.compile(loss=[wasserstein_loss, reciprocal_loss],loss_weights=[1,1], optimizer=optimizer)


    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))

        if self.divergence == 'GAN':
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(1, activation=self.last_layer_activation))

        model.summary()

        T = Input(shape=(self.latent_dim,))
        D = model(T)

        return Model(T, D)

    def train(self, epochs, batch_size=40):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        data_u = np.zeros((batch_size,self.latent_dim))
        data_sorted_x = np.zeros((10000,self.latent_dim))
        cdf_x = np.zeros((10000,self.latent_dim))

        # Generate all data once and estimate cdf
        # Generate observations and pseudo-observations
        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise
        # sort the data:
        for i in range(self.latent_dim):
            data_sorted_x[:,i] = np.sort(data[:,i])

        # calculate the proportional values of samples
        for i in range(self.latent_dim):
            cdf_x[:,i] = 1. * np.arange(len(data[:,i])) / (len(data[:,i]) - 1)

        for epoch in range(epochs):

            # ---------------------
            #  Train CODINE, Toy model
            # ---------------------

            # Generate observations and pseudo-observations
            time = np.random.normal(0, 1, (batch_size, 1))
            noise = 0.1*np.random.normal(0, 1, (batch_size, self.latent_dim))
            data_x = np.concatenate((np.sin(time),time*np.cos(time)),axis=1)+noise

            # transform data
            for i in range(self.latent_dim):
                for j in range(batch_size):
                    data_u[j,i] = cdf_x[np.argmin(np.abs(data_x[j,i]-data_sorted_x[:,i])),i]

            data_pi = np.random.uniform(0, 1, (batch_size, self.latent_dim))

            D_value_1 = self.discriminator.predict(data_u)
            D_value_2 = self.discriminator.predict(data_pi)

            if self.divergence == 'KL':
                d_loss = self.combined.train_on_batch([data_u,data_pi],[valid,valid])
                R = D_value_1
                SC = D_value_2

            elif self.divergence == 'GAN':
                d_loss = self.combined.train_on_batch([data_u,data_pi],[fake,valid])
                R = (1 - D_value_1) / D_value_1
                SC = (1 - D_value_2) / D_value_2

            elif self.divergence == 'HD':
                d_loss = self.combined.train_on_batch([data_u, data_pi], [valid, valid])
                R = 1 / (D_value_1**2)
                SC = 1 / (D_value_2**2)

            copula_estimate = R
            self_consistency = SC
            # Plot the progress
            print ("%d [D total loss : %f, Copula estimates: %f, Self-consistency mean test: %f]" % (epoch, d_loss[0], np.mean(copula_estimate), np.mean(self_consistency)))

    def test(self, test_batch = 100, test_size=1000):
        copula_density_testbatch = np.zeros((test_batch, test_size))
        data_u_testbatch = np.zeros((test_batch, test_size,self.latent_dim))
        data_pi_testbatch = np.zeros((test_batch, test_size,self.latent_dim))
        data_u = np.zeros((test_size, self.latent_dim))

        # for the quantile estimates
        data_sorted_x = np.zeros((10000, self.latent_dim))
        cdf_x = np.zeros((10000, self.latent_dim))

        # Generate all data once and estimate cdf
        # Generate observations and pseudo-observations
        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

        # sort the data:
        for i in range(self.latent_dim):
            data_sorted_x[:, i] = np.sort(data[:, i])

        # calculate the proportional values of samples
        for i in range(self.latent_dim):
            cdf_x[:, i] = 1. * np.arange(len(data[:, i])) / (len(data[:, i]) - 1)

        for M in range(test_batch):
            # Generate observations and pseudo-observations
            time = np.random.normal(0, 1, (test_size, 1))
            noise = 0.1*np.random.normal(0, 1, (test_size, self.latent_dim))
            data_x = np.concatenate((np.sin(time),time*np.cos(time)),axis=1)+noise

            # transform data
            for i in range(self.latent_dim):
                for j in range(test_size):
                    data_u[j, i] = cdf_x[np.argmin(np.abs(data_x[j, i] - data_sorted_x[:, i])), i]

            data_pi = np.random.uniform(0, 1, (test_size, self.latent_dim))

            D_value_1 = self.discriminator.predict(data_u)

            if self.divergence == 'KL':
                copula_estimate = D_value_1
            elif self.divergence == 'GAN':
                copula_estimate = (1 - D_value_1) / D_value_1
            elif self.divergence == 'HD':
                copula_estimate = 1 / ((D_value_1)**2)

            copula_density_testbatch[M, :] = np.squeeze(copula_estimate,axis=1)
            data_u_testbatch[M,:,:] = data_u
            data_pi_testbatch[M,:,:] = data_pi

        return copula_density_testbatch, data_u_testbatch, data_pi_testbatch

    def copula_gibbs_sampling(self, grid_points = 10, test_size=1000):
        # Sample from Copula using Gibbs sampling mechanism. The uv_samples variable contains the two components
        a = np.linspace(0,1,grid_points)
        uv_samples = np.zeros((test_size,self.latent_dim))
        uv_samples[0,:] = np.random.uniform(0,1,self.latent_dim) # random initialization
        for t in range(1,test_size):
            # for every component
            for i in range(self.latent_dim):
                if i == 0:
                    uv_i_vector = np.concatenate((a.reshape(-1,1),np.repeat(uv_samples[t-1,i+1:self.latent_dim],repeats = grid_points, axis = 0).reshape(-1,1)),axis=1)
                elif i>0 and i<self.latent_dim-1:
                    uv_i_vector_left = np.concatenate((np.repeat(uv_samples[t,0:i],repeats = grid_points, axis = 0).reshape(-1,1),a.reshape(-1,1)),axis=1)
                    uv_i_vector = np.concatenate((uv_i_vector_left,np.repeat(uv_samples[t-1,i+1:self.latent_dim],repeats = grid_points, axis = 0).reshape(-1,1)),axis=1)
                else:
                    uv_i_vector = np.concatenate((np.repeat(uv_samples[t,0:i],repeats = grid_points, axis = 0).reshape(-1,1),a.reshape(-1,1)),axis=1)

                disc_output = self.discriminator.predict(uv_i_vector)
                if self.divergence == 'KL':
                    copula_density_vector = disc_output
                elif self.divergence == 'GAN':
                    copula_density_vector = (1 - disc_output) / disc_output
                elif self.divergence == 'HD':
                    copula_density_vector = 1 / ((disc_output) ** 2)

                copula_density_vector = copula_density_vector/np.sum(copula_density_vector)
                # estimate the ICDF
                icdf = inverse_transform_sampling(np.squeeze(copula_density_vector), np.linspace(0,1,grid_points+1))
                unif_source = np.random.uniform(0,1)
                uv_samples[t, i] = icdf(unif_source)

        return uv_samples

    def data_sampling(self, uv_samples, grid_points = 10):
        # Generate observations and pseudo-observations
        time = np.random.normal(0, 1, (10000, 1))
        noise = 0.1 * np.random.normal(0, 1, (10000, self.latent_dim))
        data = np.concatenate((np.sin(time), time * np.cos(time)), axis=1) + noise

        xy_samples = np.zeros((10000, self.latent_dim))

        for i in range(self.latent_dim):
            hist, bin_edges = np.histogram(data[:,i], bins=grid_points, density=True)
            hist = hist / np.sum(hist)
            icdf = inverse_transform_sampling(hist, bin_edges)

            for t in range(10000):
                xy_samples[t, i] = icdf(uv_samples[t,i])

        plt.scatter(data[:, 0], data[:, 1], c="red")
        plt.scatter(xy_samples[:, 0], xy_samples[:, 1], c="blue")
        # To show the plot
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

    # Initialize dDIME
    codine = CODINE(latent_dim, divergence)
    # Train
    codine.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
    # Test
    copula_density, data_u, data_u_random = codine.test(test_batch=test_batch, test_size=test_size)
    # Sample via Gibbs, generate pseudo-observation
    uv_generated = codine.copula_gibbs_sampling(grid_points=30, test_size=10000)
    # Inverse transform sampling, get new observations, scatter plot
    xy_generated = codine.data_sampling(uv_generated, grid_points=30)

    # save on Matlab
    if save_on_matlab:
        sio.savemat('CODINE_Gibbs_Toy.mat', {'xy_generated': xy_generated, 'uv_generated': uv_generated,'c_u': copula_density, 'u': data_u, 'u_pi': data_u_random})

