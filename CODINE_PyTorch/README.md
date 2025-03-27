<div align="center">
  
# CODINE: Copula Density Neural Estimator

This folder contains the official **PyTorch** implementation of the paper.

</div>

---

# 💻 How to run the code

The codes available in the repository are developed for:
- The Gaussian copula density estimation `CODINE_Gaussian_pytorch.py`
- The Mixture of Gaussians (MoG) copula density estimation `CODINE_MoG_pytorch.py`
- The Mutual Information (MI) estimation `CODINE_MI_pytorch.py`
- The 2D toy-example generation `CODINE_Toy_pytorch.py`
- The FashionMNIST generation `CODINE_Gibbs_Fashion_pytorch.py`

Three divergence options are available to train your own CODINE model:
- KL (Kullback-Leibler)
- GAN (Generative adversarial network discriminative distance)
- HD (Hellinger distance)



To train and test CODINE on the Gaussian copula (AWGN channel, it estimates the copula density of the output y), use the following command
> python CODINE_Gaussian_pytorch.py

To change the $f$-divergence, use the following command
> python CODINE_Gaussian_pytorch.py --divergence GAN

To change the dimension of the output (and so the dimension of the copula density), and the noise correlation coefficient, use the following command
> python CODINE_Gaussian_pytorch.py --latent_dim 2 --rho 0.5

Training and testing parameters such as training epochs, batch and test sizes can be given as input
> python CODINE_Gaussian_pytorch.py --epochs 500 --batch_size 32 --test_size 10000
