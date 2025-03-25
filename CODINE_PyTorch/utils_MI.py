import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import pandas as pd
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from math import *
import json
import csv
from scipy.interpolate import interp1d
from datetime import datetime
import time


def get_uniform_from_distribution_ecdf(x, data):
  # Sort data and calculate ECDF
  sorted_data = np.sort(data)
  cdf_approx = np.searchsorted(sorted_data, x) / len(data)
  return cdf_approx


def probability_integral_transform_ecdf(samples):
    u = []
    for idx, x_i in enumerate(zip(*samples)):
        u_i = []
        sorted_x_i = np.sort(x_i)
        x_i_len = len(x_i)
        for x in x_i:
            u_i.append(np.searchsorted(sorted_x_i, x) / x_i_len)
        u.append(u_i)
    return np.transpose(np.array(u))


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true* y_pred)


def reciprocal_loss(y_true, y_pred):
    return torch.mean(torch.pow(y_true*y_pred,-1))


def my_binary_crossentropy(y_true, y_pred):
    return -torch.mean(torch.log(y_true)+torch.log(y_pred))


def logmeanexp_loss(y_pred, device="cpu"):
    eps = 1e-5
    batch_size = y_pred.size(0)
    logsumexp = torch.logsumexp(y_pred, dim=(0,))
    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)


def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


def derangement(l, device):
    o = l[:]
    while any(x == y for x, y in zip(o, l)):
        random.shuffle(l)
    return torch.Tensor(l).int().to(device)


def data_generation_mi(data_x, data_y, device="cpu"):
    der = True
    data_xy = torch.hstack((data_x, data_y))
    if der:  # Derangement
        data_y_shuffle = torch.index_select(data_y, 0, derangement(list(range(data_y.shape[0])), device))
        #ordered_derangement = [(idx + 1) % data_y.shape[0] for idx in range(data_y.shape[0])]
        #data_y_shuffle = torch.index_select(data_y, 0, torch.Tensor(ordered_derangement).int().to(device))
    else:  # Permutation
        data_y_shuffle = torch.index_select(data_y, 0, torch.tensor(np.random.permutation(data_y.shape[0])).int().to(device))

    data_x_y = torch.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y


def sample_gaussian(batch_size, latent_dim, eps, mode="gauss"):
    x = np.random.normal(0, 1, (batch_size, latent_dim))
    y = x + eps * np.random.normal(0, 1, (batch_size, latent_dim))
    if mode == "asinh":
        x = np.log(x + np.sqrt(1 + np.power(x,2)))
        y = np.log(y + np.sqrt(1 + np.power(y,2)))
    return x, y


def normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))


def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, mode="gauss", device="cpu"):
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps
    if mode == "asinh":
        x = torch.log(x + torch.sqrt(1 + torch.pow(x,2)))
        y = torch.log(y + torch.sqrt(1 + torch.pow(y,2)))
    return x.to(device), y.to(device)


def sample_distribution(rho_gauss_corr, latent_dim=20, rho=0, eps=0, df=1, batch_size=64, mode="gauss", device="cpu"):
    if mode == "gauss" or mode == "asinh":
        if rho_gauss_corr:
            x, y = sample_correlated_gaussian(dim=latent_dim, rho=rho, batch_size=batch_size, mode=mode, device=device)
        else:
            x, y = sample_gaussian(batch_size, latent_dim, eps, mode=mode)
    return x, y


def mi_to_rho(dim, mi):
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


def compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=None, buffer=None, alpha=1):
    """Compute the value of the loss and R given a certain cost function, a specific neural network and the output of
    such a neurl network. R is the ratio between joint density and product of marginals."""
    if divergence == 'KL':
        if "deranged" in architecture:
            loss, R = kl_fdime_deranged(D_value_1, D_value_2, alpha=alpha, device=device)
        else:
            loss, R = kl_fdime_e(scores, device=device)

    elif divergence == 'GAN':
        if "deranged" in architecture:
            loss, R = gan_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = gan_fdime_e(scores, device=device)

    elif divergence == 'HD':
        if "deranged" in architecture:
            loss, R = hd_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = hd_fdime_e(scores, device=device)

    elif divergence == "RKL":
        if "deranged" in architecture:
            loss, R = rkl_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = rkl_fdime_e(scores, device=device)

    elif divergence == 'MINE':
        if "deranged" in architecture:
            loss, R, buffer = mine_ma_deranged(D_value_1, D_value_2, buffer)
        else:
            loss, R, buffer = mine_ma(scores, buffer, momentum=0.9, device=device)

    elif divergence == 'SMILE':
        tau = 1.0 # np.inf
        if "deranged" in architecture:
            loss, R = smile_deranged(D_value_1, D_value_2, tau, device=device)
        else:
            loss, R = smile(scores, clip=tau, device=device)

    elif divergence == "CPC":
        if "deranged" in architecture:
            loss = torch.Tensor(0)
            R = 0
        else:
            loss, R = infonce(scores, device=device)

    elif divergence == "NWJ":
        if "deranged" in architecture:
            loss, R = nwj_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = nwj(scores, device=device)

    elif divergence == "SL":
        if "deranged" in architecture:
            loss, R = sl_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = sl_fdime_e(scores, device=device)

    return loss, R

###################### DERANGED ARCHITECTURES ########################


def kl_fdime_deranged(D_value_1, D_value_2, alpha, device="cpu"):
    """KL cost function"""
    eps = 1e-5
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = my_binary_crossentropy(valid_1, D_value_1) * alpha
    loss_2 = wasserstein_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    R = D_value_1 / alpha
    return loss, R


def gan_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """GAN cost function"""
    BCE = nn.BCELoss()
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    fake_1 = torch.zeros((batch_size_1, 1), device=device)
    loss_1 = BCE(D_value_1, fake_1)
    loss_2 = BCE(D_value_2, valid_2)
    loss = loss_1 + loss_2
    R = (1 - D_value_1) / D_value_1
    return loss, R


def hd_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """HD cost function """
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = wasserstein_loss(valid_1, D_value_1)
    loss_2 = reciprocal_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    R = 1 / (D_value_1 ** 2)
    return loss, R


def js_fgan_lower_bound_modified(D_value_1, D_value_2):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    return -1 * F.softplus(-1 * D_value_1).mean() - F.softplus(D_value_2).mean()


def smile_deranged(D_value_1, D_value_2, tau, device="cpu"):
    """SMILE cost function """
    eps = 1e-5
    D_value_2_ = torch.clamp(D_value_2, -tau, tau)  # -> il -
    dv = D_value_1.mean() - torch.log(torch.mean(torch.exp(D_value_2_)) + eps)
    js = js_fgan_lower_bound_modified(D_value_1, D_value_2)
    with torch.no_grad():
        dv_js = dv - js
    loss = -(js + dv_js)
    R = torch.exp(js + dv_js)
    return loss, R


def mine_ma_deranged(D_value_1, D_value_2, buffer, momentum=0.9, device="cpu"):
    """Mine cost function using the deranged architecture"""
    if buffer is None:
        buffer = torch.tensor(1.0)

    loss_1 = torch.mean(D_value_1)
    buffer_update = logmeanexp_loss(D_value_2, device=device).exp()
    with torch.no_grad():
        second_term = logmeanexp_loss(D_value_2, device=device)
        buffer_new = buffer * momentum + buffer_update * (1-momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new
    third_term_grad = buffer_update / buffer_new
    loss = -(loss_1 - second_term - third_term_grad + third_term_no_grad)
    R = torch.exp(-loss)
    return loss, R, buffer_update


def rkl_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """Reverse KL cost function"""
    eps = 1e-5
    loss_1 = torch.mean(torch.pow(D_value_1 + eps, -1))
    loss_2 = torch.mean(torch.log(D_value_2 + eps))
    loss = loss_1 + loss_2
    return loss, D_value_1


def sl_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    eps = 1e-5
    loss_1 = torch.mean(D_value_1)
    loss_2 = torch.mean(torch.log(D_value_2 + eps) - D_value_2)
    R = (1-D_value_1)/D_value_1
    return loss_1-loss_2, R


def tuba_deranged(D_value_1, D_value_2, log_baseline=None):
    if log_baseline is not None:
        D_value_1 -= log_baseline[:, None]
        D_value_2 -= log_baseline[:, None]
    joint_term = D_value_1.mean()
    marg_term = logmeanexp_loss(D_value_2).exp()
    return -(1. + joint_term - marg_term)


def nwj_deranged(D_value_1, D_value_2, device="cpu"):
    loss = tuba_deranged(D_value_1 - 1., D_value_2 - 1)
    R = torch.exp(-loss)
    return loss, R

#################################### CONCAT - SEPARABLE ARCHITECTURES ###########################################


def logmeanexp_diag(x, device='cpu'):
    batch_size = x.size(0)
    eps = 1e-5
    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float() + eps).to(device)


def logmeanexp_loss(y_pred, device="cpu"):
    eps = 1e-5
    batch_size = y_pred.size(0)
    logsumexp = torch.logsumexp(y_pred, dim=(0,))
    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)


def logmeanexp_nodiag(x, dim=None, device='cpu'):
    eps = 1e-5
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)).to(device), dim=dim)
    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba(scores, log_baseline=None, device="cpu"):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    joint_term = scores.diag().mean()
    marg_term = logmeanexp_nodiag(scores, device=device).exp()
    return -(1. + joint_term - marg_term)


def nwj(scores, device="cpu"):
    loss = tuba(scores - 1., device=device)
    R = torch.exp(-loss)
    return loss, R


def infonce(scores, device="cpu"):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    mi = torch.tensor(scores.size(0), device=device).float().log() + nll
    mi = mi.mean()
    R = torch.exp(mi)
    return -mi, R


def kl_fdime_e(scores, device="cpu"):
    eps = 1e-7
    scores_diag = scores.diag()
    n = scores.size(0)
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device)

    loss_1 = -torch.mean(torch.log(scores_diag + eps))
    loss_2 = torch.sum(scores_no_diag) / (n*(n-1))
    loss = loss_1 + loss_2
    return loss, scores_diag


def gan_fdime_e(scores, device="cpu"):
    eps = 1e-5
    batch_size = scores.size(0)
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag*torch.eye(batch_size, device=device) + torch.eye(batch_size, device=device)
    R = (1 - scores_diag) / scores_diag
    loss_1 = torch.mean(torch.log(torch.ones(scores_diag.shape, device=device) - scores_diag + eps))
    loss_2 = torch.sum(torch.log(scores_no_diag + eps)) / (batch_size*(batch_size-1))
    return -(loss_1+loss_2), R


def hd_fdime_e(scores, device="cpu"):
    eps = 1e-5
    Eps = 1e7
    scores_diag = scores.diag()
    n = scores.size(0)
    scores_no_diag = scores + Eps * torch.eye(n, device=device)
    loss_1 = torch.mean(scores_diag)
    loss_2 = torch.sum(torch.pow(scores_no_diag, -1))/(n*(n-1))
    loss = -(2 - loss_1 - loss_2)
    return loss, 1 / (scores_diag**2)


def js_fgan_lower_bound(f):
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def smile(f, clip=None, device="cpu"):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1), device=device)
    dv = f.diag().mean() - z
    js = js_fgan_lower_bound(f)
    with torch.no_grad():
        dv_js = dv - js
    loss = -(js + dv_js)
    R = torch.exp(js + dv_js)
    return loss, R


def mine_ma(f, buffer=None, momentum=0.9, device="cpu"):
    buffer = None
    if buffer is None:
        buffer = torch.tensor(1.0)
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f, device=device).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f, device=device)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new
    third_term_grad = buffer_update / buffer_new
    loss = -(first_term - second_term - third_term_grad + third_term_no_grad)
    R = torch.exp(-loss)
    return loss, R, buffer_update


def rkl_fdime_e(scores, device="cpu"):
    eps = 1e-5
    n = scores.size(0)
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device) + torch.eye(n, device=device)
    loss_1 = torch.mean(torch.pow(scores_diag + eps, -1))  #+ 1e-5
    loss_2 = torch.sum(torch.log(scores_no_diag + eps))/(n*(n-1))
    loss = loss_1 + loss_2
    return loss, scores_diag


def sl_fdime_e(scores, device="cpu"):
    eps = 1e-7
    n = scores.size(0)
    scores = scores + eps
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device) + torch.eye(n, device=device)
    loss_1 = torch.mean(scores_diag)
    loss_2 = torch.sum(torch.log(scores_no_diag) - (scores_no_diag-torch.eye(n, device=device)))/(n*(n-1))
    R = (1-scores_diag)/scores_diag
    return loss_1-loss_2, R

#####################################################################################################


def plot_staircases(staircases, proc_params, opt_params, latent_dim):
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c',
        'deranged_copula': '#9467bd',
        'joint_copula': 'darkorchid',
        'separable_copula': 'teal'
    }
    now = datetime.now()
    date_to_print = now.strftime("%d_%m_%Y__%H_%M")
    n_divergences = len(proc_params["divergences"])
    n_architectures = len(proc_params['architectures'])
    fig, sbplts = plt.subplots(len(proc_params["modes"]), n_divergences, figsize=(4 * n_divergences, 4 * len(proc_params["modes"])))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    for idx, mode in enumerate(proc_params["modes"]):
        mode_sbplt = sbplts[idx]
        i = 0
        if n_divergences > 1:
            for divergence in proc_params['divergences']:
                mode_sbplt[i].plot(range(tot_len_stairs), np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                            linewidth=1, c='k', linestyle="dashed")
                for architecture in proc_params['architectures']:
                    if divergence == "CPC" and "deranged" in architecture:
                        pass
                    else:
                        fDIME_training_staircase_smooth = pd.Series(staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                        sm = mode_sbplt[i].plot(range(tot_len_stairs), staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                                 linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                        mode_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture, linewidth=1, c=sm.get_color())
                mode_sbplt[i].plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI", linewidth=1, c='k')
                if i==0:
                    mode_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
                if divergence=="GAN" or divergence=="NWJ":
                    mode_sbplt[i].legend(loc="best", fontsize=10)
                mode_sbplt[i].set_xlabel('Steps', fontsize=18)
                if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                    mode_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
                elif divergence in ["NWJ"]:
                    mode_sbplt[i].set_title("NWJ-{}".format(mode), fontsize=20)
                else:
                    mode_sbplt[i].set_title(divergence, fontsize=20)
                mode_sbplt[i].set_xlim([0, tot_len_stairs])
                mode_sbplt[i].set_ylim([0, proc_params['levels_MI'][-1]+2])
                i += 1
        else:
            divergence = proc_params['divergences'][0]
            if divergence == "CPC":
                mode_sbplt.plot(range(tot_len_stairs),
                                    np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                                    linewidth=1, c='k', linestyle="dashed")
            for architecture in proc_params['architectures']:
                if divergence == "CPC" and "deranged" in architecture:
                    pass
                else:
                    fDIME_training_staircase_smooth = pd.Series(
                        staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                        span=200).mean()
                    sm = mode_sbplt.plot(range(tot_len_stairs), staircases[
                        f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                             linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    mode_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                        linewidth=1, c=sm.get_color())
            mode_sbplt.plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI",
                                linewidth=1, c='k')
            mode_sbplt.set_ylabel('MI [nats]', fontsize=18)
            mode_sbplt.legend(loc="best")
            mode_sbplt.set_xlabel('Steps', fontsize=18)
            if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                mode_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
            elif divergence in ["NWJ"]:
                mode_sbplt.set_title("NWJ-{}".format(mode), fontsize=20)
            else:
                mode_sbplt.set_title(divergence, fontsize=20)
            mode_sbplt.set_xlim([0, tot_len_stairs])
            mode_sbplt.set_ylim([0, proc_params['levels_MI'][-1] + 2])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}_arc{}_date{}.svg".format(latent_dim, opt_params['batch_size'], proc_params["architectures"][0], date_to_print))


def save_time_dict(time_dict, latent_dim, batch_size, proc_params, scenario):
    with open("Results/Stairs/time_dictionary_d{}_bs{}_arc{}_scen{}.json".format(latent_dim, batch_size, proc_params["architectures"][0], scenario), "w") as fp:
        json.dump(time_dict.copy(), fp)


def save_dict_lists_csv(path, dictionary):
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))

