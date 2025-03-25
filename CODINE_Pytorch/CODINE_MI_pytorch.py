from classes_MI import *
from utils_MI import *
from multiprocessing import Pool, Manager


def train_with_config(config):
    architecture = config["architecture"]
    for mode in proc_params['modes']:
        for divergence in proc_params['divergences']:
            if divergence == "CPC" and ("deranged" in architecture):
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = 0
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = 0
            else:
                start_time = time.time()
                copula_mi = CODINE(proc_params, divergence, architecture, mode)
                mi_estimates = np.zeros(proc_params['len_step'] * len(proc_params['levels_MI']))
                for idx, level_MI in enumerate(proc_params['levels_MI']):
                    copula_mi.update_SNR_or_rho(level_MI)
                    copula_mi_training_estimates_tmp = copula_mi.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'])
                    mi_estimates[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = copula_mi_training_estimates_tmp
                end_time = time.time()
                time_dict[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}_{proc_params["latent_dim"]}'] = (float(end_time) - float(start_time)) / 60
                staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = mi_estimates


if __name__ == '__main__':

    batch_sizes = [32]
    architectures_list = [["joint", "joint_copula"]]
    for batch_size in batch_sizes:
        for architectures in architectures_list:
            proc_params = {
                'levels_MI': [2,4,6,8,10],
                'len_step': 4000,
                'alpha': 1,
                'latent_dim': 2,
                'divergences': ["KL", "HD"],
                'architectures': architectures,
                'tot_len_stairs': 20000,
                'modes': ["gauss","asinh"],
                'rho_gauss_corr': False,
                'device': "cpu"
            }
            opt_params = {
                'batch_size': batch_size
            }

            training_configs = [
                {"architecture": architectures[0]},
                {"architecture": architectures[1]},
            ]

            manager = Manager()
            staircases = manager.dict()
            time_dict = manager.dict()
            pool = Pool(processes=2)
            pool.map(train_with_config, training_configs)
            pool.close()
            plot_staircases(staircases, proc_params, opt_params, proc_params['latent_dim'])
            save_time_dict(time_dict, proc_params['latent_dim'], opt_params["batch_size"], proc_params, "staircase")

