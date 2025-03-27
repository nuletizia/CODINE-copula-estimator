from utils_MI import *


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(200, 100),
            nn.ReLU(0.2),
            nn.Dropout(0.2),
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
            "RKL": nn.Softplus(),
            "HD": nn.Softplus(),
            "MINE": nn.Identity(),
            "GAN_DIME": nn.Sigmoid(),
            "SL": nn.Sigmoid(),
            "SMILE": nn.Identity(),
            "NWJ": nn.Identity()
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


class ConcatCritic(nn.Module):
    def __init__(self, dim, hidden_dim, layers, activation, divergence):
        super(ConcatCritic, self).__init__()
        self._f = mlp(dim, hidden_dim, 1, layers, activation)
        if divergence == "GAN" or divergence == "SL":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD" or divergence == "RKL":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        batch_size = x.size(0)
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        scores = self._f(xy_pairs)
        out = torch.reshape(scores, [batch_size, batch_size]).t()
        out = self.last_activation(out)
        return out


class SeparableCritic(nn.Module):
    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, divergence, mode):
        super(SeparableCritic, self).__init__()
        if mode == "swiss":
            self._g = mlp(2*dim, hidden_dim, embed_dim, layers, activation)
        else:
            self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)
        if divergence == "GAN" or divergence == "SL":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD" or divergence == "RKL":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return self.last_activation(scores)


class CODINE():
    def __init__(self, proc_params, divergence, architecture, mode):
        self.latent_dim = proc_params['latent_dim']
        self.divergence = divergence
        self.mode = mode
        self.device = proc_params['device']
        self.architecture = architecture
        self.alpha = proc_params['alpha']
        self.rho_gauss_corr = proc_params['rho_gauss_corr']
        self.rho = None
        self.eps = None
        self.df = None
        if mode == "swiss":
            if self.architecture == "deranged" or self.architecture == "deranged_copula":
                simple_net = Net(3, 1)
                self.discriminator = CombinedNet(simple_net, self.divergence)
            elif self.architecture == "joint" or self.architecture == "joint_copula":
                self.discriminator = ConcatCritic(3, 256, 2, 'relu', self.divergence).to(self.device)
            elif self.architecture == 'separable' or self.architecture== "separable_copula":
                self.discriminator = SeparableCritic(self.latent_dim, 256, 32, 2, 'relu', self.divergence, self.mode).to(self.device)
        else:
            if self.architecture == "deranged" or self.architecture == "deranged_copula":
                simple_net = Net(2 * self.latent_dim, 1)
                self.discriminator = CombinedNet(simple_net, self.divergence)
            elif self.architecture == "joint" or self.architecture == "joint_copula":
                self.discriminator = ConcatCritic(2 * self.latent_dim, 256, 2, 'relu', self.divergence).to(self.device)
            elif self.architecture == 'separable' or self.architecture== "separable_copula":
                self.discriminator = SeparableCritic(self.latent_dim, 256, 32, 2, 'relu', self.divergence, self.mode).to(self.device)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.002)

    def update_SNR(self, SNR):
        self.EbN0 = SNR
        self.eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))

    def update_rho(self, rho):
        self.rho = rho

    def update_SNR_or_rho(self, level_MI):
        if self.rho_gauss_corr:
            rho = mi_to_rho(self.latent_dim, level_MI)
            self.update_rho(rho)
        else:
            SNR = 10 * np.log10(np.exp(2 * level_MI / self.latent_dim) - 1)
            self.update_SNR(SNR)

    def update_eps_unif(self, eps):
        self.eps = eps

    def update_rho_stud(self, rho):
        self.rho = rho

    def update_df(self, df):
        self.df = df

    def transform_sampling(self, x, y):
        if self.mode == "gauss":
            data_u = torch.tensor(phi(x, 0, 1)).float().to(self.device)
            data_v = torch.tensor(phi(y, 0, np.sqrt(1 + self.eps ** 2))).float().to(self.device)
        elif self.mode == "cubic":
            data_u = torch.tensor(phi(x, 0, 1)).float().to(self.device)
            data_v = torch.tensor(probability_integral_transform_ecdf(y)).float().to(self.device)
        elif self.mode == "half-cube" or self.mode == "asinh" or self.mode == "uniform" or self.mode == "swiss" or self.mode == "student":
            data_u = torch.tensor(probability_integral_transform_ecdf(x)).float().to(self.device)
            data_v = torch.tensor(probability_integral_transform_ecdf(y)).float().to(self.device)
        return data_u, data_v

    def train(self, epochs, batch_size=40, random_seed=0, verbose=True):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.discriminator.train()
        mi_training_estimates = []
        for epoch in range(epochs):
            x, y = sample_distribution(self.rho_gauss_corr, latent_dim=self.latent_dim, rho=self.rho, eps=self.eps,
                                       df=self.df, batch_size=batch_size, mode=self.mode, device=self.device)
            self.optimizer.zero_grad()
            if "copula" in self.architecture:
                data_u, data_v = self.transform_sampling(x, y)
            else:
                if not self.rho_gauss_corr:
                    data_u = torch.tensor(x).float()
                    data_v = torch.tensor(y).float()
                else:
                    data_u = x
                    data_v = y
            if "deranged" in self.architecture:
                data_uv, data_u_v = data_generation_mi(data_u, data_v, device=self.device)
                D_value_1, D_value_2 = self.discriminator(data_uv, data_u_v)
                loss, R = compute_loss_ratio(self.divergence, self.architecture, D_value_1=D_value_1,
                                             D_value_2=D_value_2,
                                             scores=None, buffer=None, alpha=self.alpha, device=self.device)
            else:
                scores = self.discriminator(data_u, data_v)
                loss, R = compute_loss_ratio(self.divergence, self.architecture, D_value_1=None, D_value_2=None,
                                             scores=scores, buffer=None, alpha=self.alpha, device=self.device)

            mi_copula_estimate = torch.mean(torch.log(R))

            loss.backward()
            self.optimizer.step()

            mi_training_estimates.append(mi_copula_estimate.detach().numpy())
            if verbose and epoch % 1000 == 0:
                print(f"{epoch} [Mode: {self.mode}, Divergence: {self.divergence}, Architecture: {self.architecture}, "
                      f"Total loss : {loss.item():.4f}, MI now: {mi_copula_estimate:.4f}")

        return mi_training_estimates


