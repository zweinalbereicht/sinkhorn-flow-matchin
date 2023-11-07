import torch
import torch.nn as nn
import torch.distributions as D
from sinkhorn import Sinkhorn

class GenModel:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        raise NotImplementedError

    def save(self, epoch_num=None):
        pass

    def load(self, epoch_num=None):
        pass

# generates 2*n_dim data from the optimal coupling by convoluting with a gaussian for each bin
class SinkhornModel():
    def __init__(self,sinkhorn : Sinkhorn):
        # super().__init__()
        self.bins = sinkhorn.data.bins
        self.bin_size = sinkhorn.data.bin_size
        self.push_forward = sinkhorn.push_forward

        # watch out for discrepancy in size of bins and push_forward
        nb_bins = len(self.push_forward)

        self.weights = torch.zeros_like(self.push_forward).flatten()
        self.means = torch.zeros(len(self.weights),2)
        self.covs = torch.zeros(len(self.weights),2,2)
        assert self.weights.shape[0] == self.means.shape[0] == self.covs.shape[0]

        bin_centers = (self.bins[1:] - self.bin_[:-1]) / 2
        for i,b1 in enumerate(bin_centers):
            for j,b2 in enumerate(bin_centers):
                self.weights[i + nb_bins * j] = self.push_forward[i,j]
                self.means[i + nb_bins * j] = torch.tensor([b1,b2])
                self.covs[i + nb_bins * j] = torch.eye(2) * self.bin_size ** 2 # arbitrary choice here and hardcoded for dimension 1 -> 1


        component_distributions = D.Independent(D.MultivariateNormal(self.means,
                                                                     self.covs), 0)
        mixture_distribution = D.Categorical(self.weights)
        self.gmm = D.MixtureSameFamily(mixture_distribution,
                                       component_distributions)

    def log_prob(self, all_x):
        log_prob = self.gmm.log_prob(all_x)
        return log_prob

    def sample(self, n_samples):
        sample = self.gmm.sample((n_samples,))
        log_prob = self.gmm.log_prob(sample)
        return sample, log_prob

class FMModel:
    def __init__(self, score_network,
                 score_network_opt, score_network_scheduler,
                 rho_0_dist,
                 device=torch.device('cuda:0')):
        self.score_network = score_network
        self.score_network_opt = score_network_opt
        self.score_network_scheduler = score_network_scheduler
        self.rho_0_dist = rho_0_dist
        self.device = device

    def get_flow_matching_loss(self, x0, x1, sigma=0.01):
        """
        Arguments:
            x0: torch.Tensor of shape (batch_size, n_dim)
            x1: torch.Tensor of shape (batch_size, n_dim)
            sigma: float
        """
        ts = ((torch.rand((x0.shape[0], 1), device=self.device)
               * (1 - 1e-4))
              + 1e-4)
        mu = ts * x1 + (1 - ts) * x0
        x = torch.randn_like(x1) * sigma + mu
        loss = nn.MSELoss()((x1 - x0),
                            self.score_network(x, ts))
        return loss

    def sample(self, n_samples, n_time_steps=100):
        """
        Arguments:
            n_samples: int
            n_time_steps: int
        Returns:
            all_x_t: torch.Tensor of shape (n_samples, n_time_steps, n_dim)
        """
        all_x_t = []
        x_t, _ = self.rho_0_dist.sample(n_samples)
        all_t = torch.linspace(0, 1, n_time_steps, device=self.device)
        all_dt = torch.diff(all_t)

        for (dt, t) in zip(all_dt, all_t[:-1]):
            drift = self.score_network(x_t, torch.ones(
                x_t.shape[0], 1, device=self.device).float() * t)
            x_t = x_t + drift * dt
            all_x_t.append(x_t)

        all_x_t = torch.stack(all_x_t, dim=1)
        return all_x_t

    def save(self, epoch_num=None):
        pass
