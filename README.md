<div align="center">
  
# CODINE: Copula Density Neural Estimator

[Nunzio A. Letizia](https://scholar.google.com/citations?user=v50jRAIAAAAJ&hl=en), [Nicola Novello](https://scholar.google.com/citations?user=4PPM0GkAAAAJ&hl=en), [Andrea M. Tonello](https://scholar.google.com/citations?user=qBiseEsAAAAJ&hl=en)<br />

</div>

Official repository of the paper "Copula Density Neural Estimation".

> CODINE is a neural copula density estimator that estimates any copula density by maximizing a variational lower bound on the $f$-divergence.

<div align="center">

[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/nicolaNovello/CODINE-copula-estimator/blob/main/LICENSE)
[![Hits](https://hits.sh/github.com/nicolaNovello/CODINE-copula-estimator.svg?label=Visitors&color=30a704)](https://hits.sh/github.com/nicolaNovello/CODINE-copula-estimator/)

</div>

---

# 📈 Important results from our paper

Please refer to the paper to have a precise description of all the results.

## Copula density estimation

The main purpose of CODINE is to estimate the copula density. In the following, we present three results of copula density estimation in three different settings.

### Gaussian

<img src="Figures/teaser_gaussian.jpg" width=600/>

### Toy scenario

<img src="Figures/teaser_toy.jpg" width=600/>

### Mixture of Gaussians

<img src="Figures/teaser_MoG.png"/>

## Mutual information estimation

CODINE can be adapted to be used as a MI estimator.

### Gaussian

<img src="Figures/teaser_MI_gauss.jpg" width=600/>

### Gaussian with asinh transformation

<img src="Figures/teaser_MI_asinh.jpg" width=600/>

## Data generation

Once the copula density is estimated using CODINE, we show how to generate data sampling the estimated copula with Gibbs sampling.

### MNIST digits

We show the generated digits and the architecture of the decoder we use for generation.

<img src="Figures/teaser_digits_generation.jpg" width=600/>

### FashionMNIST

Since the data is generated using a decoder, we compare the data generated feeding the decoder with the sampling of CODINE's estimated copula (referred to as codine generation) and a random sampling of a uniform distribution (random generation). 

<img src="Figures/teaser_fashion_random.jpg" width=600/>

We compare the generation using different dimensions of the latent space.

<img src="Figures/teaser_fashion_dims.jpg" width=600/>


---

# 💻 How to run the code

Depending on your preferred library for neural implementatins, please refer to the corresponding folder:
- `CODINE_Keras` when using Keras (for a fast shortcut, click [here](https://github.com/nicolaNovello/CODINE-copula-estimator/tree/main/CODINE_Keras))
- `CODINE_PyTorch` when using PyTorch (for a fast shortcut, click [here](https://github.com/nicolaNovello/CODINE-copula-estimator/tree/main/CODINE_PyTorch))

---

## 🤓 General description

The paper presents a copula density estimation method, denoted as CODINE.
CODINE is a neural network trained to estimate the copula density (and thus the pdf) associated to any data. By design, it works with pseudo-observations (data in the uniform probability space). It can be used for:
- Density estimation
- Dependence measures
- Mutual information estimation
- Data generation
- More...

The copula density is estimated by maximizing the following objective function with respect to $T$:

$$\mathcal{J}_ {f}(T) = \mathbb{E}_ {\mathbf{u} \sim c_{U}(\mathbf{u})}\biggl[T\bigl(\mathbf{u}\bigr)\biggr] - \mathbb{E}_ {\mathbf{u} \sim \pi_{U}(\mathbf{u})}\biggl[f^*\biggl(T\bigl(\mathbf{u}\bigr)\biggr)\biggr] ,$$

where $c_U$ is the copula density and $\pi_ {U}$ is a multivariate uniform density. Given the optimal $\hat{T}$ that maximizes $\mathcal{J}_ {f}(T)$, the copula estimate is obtained as

$$c_U(\mathbf{u}) = \bigl(f^{*}\bigr)^{\prime} \bigl(\hat{T}(\mathbf{u})\bigr).$$

---

## 📝 References 

If you use the code for your research, please cite our paper:
```
@article{letizia2022copula,
  title={Copula density neural estimation},
  author={Letizia, Nunzio A and Tonello, Andrea M},
  journal={arXiv preprint arXiv:2211.15353},
  year={2022}
}
```
## 📋 Acknowledgments

The implementation is based on / inspired by:

- [https://github.com/tonellolab/fDIME](https://github.com/tonellolab/fDIME)

---

## 📧 Contact

[nunzio.letizia@aau.at](nunzio.letizia@aau.at)

[nicola.novello@aau.at](nicola.novello@aau.at)
