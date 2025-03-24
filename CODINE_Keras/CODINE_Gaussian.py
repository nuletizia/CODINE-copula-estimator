from __future__ import print_function, division

from keras.layers import Input, Dense, GaussianNoise, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

K.clear_session()
import numpy as np
import scipy.io as sio
import argparse
from math import *
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
    def __init__(self, latent_dim, EbN0, divergence = 'KL', rho = 0):

        # Input shape
        self.latent_dim = latent_dim
        self.EbN0 = EbN0
        self.divergence = divergence # type of f-divergence to use for training and estimation
        self.last_layer_activation = 'softplus'

        # Noise std based on EbN0 in dB
        eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))

        self.eps = eps
        self.rho = rho

        optimizer = Adam(0.002, 0.5)

        # Build and compile the discriminator (CODINE has a discriminative formulation)
        self.discriminator = self.build_discriminator()

        v = Input(shape=(self.latent_dim,))
        v_random = Input(shape=(self.latent_dim,))

        # The discriminator takes as input joint or marginal vectors
        d_v = self.discriminator(v)
        d_v_random = self.discriminator(v_random)

        # Train the discriminator
        self.combined = Model([v, v_random], [d_v,d_v_random])

        # choose the loss function based on the f-divergence type
        if self.divergence == 'KL':
            self.combined.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[1,1], optimizer=optimizer)
        elif self.divergence == 'GAN':
            self.combined.compile(loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[1,1], optimizer=optimizer)
        elif self.divergence == 'HD':
            self.combined.compile(loss=[wasserstein_loss, reciprocal_loss],loss_weights=[1,1], optimizer=optimizer)
        elif self.divergence == 'MINE':
            self.combined.compile(loss=[wasserstein_loss, logsumexp_loss],loss_weights=[-1,1], optimizer=optimizer)


    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.3))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))

        if self.divergence == 'GAN':
            model.add(Dense(1, activation='sigmoid'))
        elif self.divergence == 'MINE':
            model.add(Dense(1))
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

        rho = self.rho
        Sigma_N = np.zeros((self.latent_dim, self.latent_dim))

        # Define the cov matrix
        for i in range(self.latent_dim - 1):
            Sigma_N[i, i + 1] = 1
            Sigma_N[i + 1, i] = 1
        Sigma_N = np.eye(self.latent_dim) + rho * Sigma_N

        for epoch in range(epochs):

            # ---------------------
            #  Train CODINE, Gaussian model
            # ---------------------

            # Sample noise and generate a batch
            channel_input = np.random.normal(0, 1, (batch_size, self.latent_dim))
            eps = self.eps
            channel_output = channel_input + eps * np.random.multivariate_normal(np.zeros((self.latent_dim,)),Sigma_N,batch_size)

            # Get pseudo-observations via transform sampling
            data_v = phi(channel_output,0,np.sqrt(1+eps**2))
            data_v_random = np.random.uniform(0,1,(batch_size, self.latent_dim))
            D_value_1 = self.discriminator.predict(data_v)
            D_value_2 = self.discriminator.predict(data_v_random)

            if self.divergence == 'KL':
                d_loss = self.combined.train_on_batch([data_v,data_v_random],[valid,valid])
                R = D_value_1
                SC = D_value_2

            elif self.divergence == 'GAN':
                d_loss = self.combined.train_on_batch([data_v,data_v_random],[fake,valid])
                R = (1 - D_value_1) / D_value_1
                SC = (1 - D_value_2) / D_value_2

            elif self.divergence == 'HD':
                d_loss = self.combined.train_on_batch([data_v, data_v_random], [valid, valid])
                R = 1 / (D_value_1**2)
                SC = 1 / (D_value_2**2)

            elif self.divergence == 'MINE':
                d_loss = self.combined.train_on_batch([data_v, data_v_random], [valid, valid])
                R = np.exp(-d_loss[0])
                SC = np.exp(-d_loss[2])

            copula_estimate = R
            self_consistency = SC
            # Plot the progress
            print ("%d [D total loss : %f, Copula estimates: %f, Self-consistency mean test: %f]" % (epoch, d_loss[0], np.mean(copula_estimate), np.mean(self_consistency)))


    def test(self, test_batch = 100, test_size=1000):
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
            data_v = phi(channel_output,0,np.sqrt(1+eps**2))
            data_v_random = np.random.uniform(0,1,(test_size, self.latent_dim))

            D_value_1 = self.discriminator.predict(data_v)
            D_value_2 = self.discriminator.predict(data_v_random)

            if self.divergence == 'KL':
                copula_estimate = D_value_1
                SC = D_value_2 # should be one
            elif self.divergence == 'GAN':
                copula_estimate = (1 - D_value_1) / D_value_1
                SC = (1 - D_value_2) / D_value_2 # should be one
            elif self.divergence == 'HD':
                copula_estimate = 1 / ((D_value_1)**2)
                SC = 1 / ((D_value_2)**2) # should be one

            elif self.divergence == 'MINE':
                copula_estimate = np.exp(D_value_1) # dummy
                SC = np.exp(D_value_2) # dummy


            copula_density_testbatch[M, :] = np.squeeze(copula_estimate,axis=1)
            c_test_testbatch[M, :] = np.squeeze(SC,axis=1)
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

        # plot the Gaussian copula density estimate for SNR = 0
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(data_v[0, :, 0], data_v[0, :, 1], copula_density[0,:], color="green")
        plt.title("Gaussian copula density estimate with CODINE")

        # To show the plot
        plt.show()

        del codine
        j = j+1

    # save on Matlab
    if save_on_matlab:
        sio.savemat('CODINE_Gaussian.mat', {'SNR': SNR_dB,'c_u': copula_density_total, 'c_test':c_test_total, 'u': data_u_total, 'v': data_v_total})
