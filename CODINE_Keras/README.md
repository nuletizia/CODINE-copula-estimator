<div align="center">
  
# CODINE: Copula Density Neural Estimator

</div>

This folder contains the official **Keras** implementation of the paper.

---

# 📈 Results from our paper

The codes available in the repository are developed for the Gaussian copula density estimation and for the 2d toy-example. It can be extended to any data using the transform sampling functions available in the latter.

<img src="https://github.com/nuletizia/CODINE-copula-estimator/blob/main/teaser_gaussian.jpg" width=400>

---

# 💻 How to run the code

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

