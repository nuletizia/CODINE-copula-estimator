# CODINE-copula-estimator
Copula density neural estimation

This repository contains the official implementation of the following paper:

Copula Density Neural Estimation - under submission

If you use the repository for your experiments, please cite the paper.

<img src="https://github.com/nuletizia/CODINE-copula-estimator/blob/main/teaser_gaussian.jpg" width=800>


The paper presents a density estimation method based on the copula, denoted as CODINE.
CODINE is a neural network trained to estimate the copula density (and thus the pdf) associated to any data. By design, it works with pseudo-observations (data in the uniform probability space). It can be used for:
- Density estimation
- Dependence measures
- Mutual information estimation
- Data generation
- More..

The codes available in the repository are developed for the Gaussian copula density estimation and for the 2d toy-example. It can be extended to any data using the transform sampling functions available in the latter.

<img src="https://github.com/nuletizia/CODINE-copula-estimator/blob/main/teaser_toy.jpg" width=800>

Three divergence options are available to train your own CODINE model:
- KL (Kullback-Leibler)
- GAN (Generative adversarial network discriminative distance)
- HD (Hellinger distance)

To train and test CODINE on the Gaussian copula (AWGN channel, it estimates the copula density of the output y), use the following command
> python CODINE_Gaussian.py

To change the f-divergence, use the following command
> python CODINE_Gaussian.py --divergence GAN

To change the dimension of the output (and so the dimension of the copula density), and the noise correlation coefficient, use the following command
> python CODINE_Gaussian.py --latent_dim 2 --rho 0.5

Training and testing parameters such as training epochs, batch and test sizes can be given as input
> python CODINE_Gaussian.py --epochs 500 --batch_size 32 --test_size 10000

To train and test CODINE on the 2d toy example, use the following command (arguments can be added as in the Gaussian case)
> python CODINE_Toy.py
