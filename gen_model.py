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

class CouplingModel():
    def __init__(self,bins_x,bins_y,coupling) -> None:
        """
        Constructs a categorical distribution to sample form a coupling with support (bins_x x bins_y)
        Importantly, we require that Probability(bins_x[i],bins_y[j])=coupling[i,j]
        Watch out, bins_x and bins_y are bin centers already !
        """
        self.bins_x=bins_x
        self.bins_y=bins_y
        self.coupling = coupling

        self.weights = self.coupling.clone().flatten()
        self.samples = torch.zeros(len(self.weights),2)

        k=0
        for b1 in self.bins_x:
            for b2 in self.bins_y:
                self.samples[k] = torch.tensor([b1,b2])
                k+=1

        print(len(self.weights),len(self.samples),len(self.bins_x))
        # watch out for empty weights which can arise
        mask = torch.clone(self.weights)>1e-10
        self.weights = self.weights[mask]
        self.samples = self.samples[mask]

        # these have turned out to be too slow, let's just use a big catgorical
        self.categorical_distribution =D.Categorical(self.weights)

    def sample(self, n_samples):
        idx_sample = self.categorical_distribution.sample((n_samples,))
        return self.samples[idx_sample]




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
        assert self.weights.shape[0] == self.means.shape[0]

        bin_centers = (self.bins[1:] + self.bins[:-1]) / 2
        for i,b1 in enumerate(bin_centers):
            for j,b2 in enumerate(bin_centers):
                self.weights[i + nb_bins * j] = self.push_forward[i,j]
                self.means[i + nb_bins * j] = torch.tensor([b1,b2])

        # watch out for empty weights which can arise
        mask = torch.clone(self.weights)>0
        self.weights = self.weights[mask]
        self.means = self.means[mask]

        # these have turned out to be too slow, let's just use a big catgorical
        self.categorical_distribution =   D.Categorical(self.weights)

    def sample_categorical(self, n_samples):
        idx_sample = self.categorical_distribution.sample((n_samples,))
        return self.means[idx_sample]
        return sample

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
        x_t = self.rho_0_dist.sample(torch.Size([n_samples,1]))
        all_t = torch.linspace(0, 1, n_time_steps, device=self.device)
        all_dt = torch.diff(all_t)
        all_x_t.append(x_t)
        for (dt, t) in zip(all_dt, all_t[:-1]):
            drift = self.score_network(x_t, torch.ones(
                x_t.shape[0], 1, device=self.device).float() * t)
            x_t = x_t + drift * dt
            all_x_t.append(x_t)

        all_x_t = torch.stack(all_x_t, dim=1)
        return all_t,all_x_t

    def save(self, epoch_num=None):
        pass
