import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import ot
from sinkhorn import Sinkhorn
from torch.utils.tensorboard import SummaryWriter
from utils import langevin

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


# generate coupling from data_samples
def generate_coupling(bins_x,bins_y,samples_x,samples_y,method="ot",reg=1e-3):
    """
    generated a coupling from samples, using POT builtin methods
    method : "product' ,"ot", "sinkhorn"


    returns:
    h_x : weights of the histogram of samples_x
    h_y : weights of the histogram of samples_y
    bins_x : centered bins of the histogram of samples_x
    bins_y : centered bins of the histogram of samples_y
    coupling : torch.tensor
    distance : Sum dij * pij
    """
    h_x = np.histogram(samples_x,bins=bins_x,density=True)[0]
    h_y = np.histogram(samples_y,bins=bins_y,density=True)[0]
    bins_x_t = torch.tensor(( bins_x[1:] + bins_x[:-1] )/ 2).unsqueeze(-1)
    bins_y_t = torch.tensor((bins_y[1:] + bins_y[:-1] ) /  2 ).unsqueeze(-1)
    cost_matrix = (bins_x_t - bins_y_t.T) ** 2
    cost_matrix_array = cost_matrix.numpy()
    if method=="product":
        coupling = (torch.tensor(h_x).unsqueeze(-1) * torch.tensor(h_y).unsqueeze(-1).T).numpy()
        coupling = coupling/coupling.sum()
    elif method=="ot":
        coupling = ot.emd(h_x,h_y,cost_matrix_array)
        coupling = coupling/coupling.sum()
    elif method=="sinkhorn":
        coupling = ot.sinkhorn(h_x,h_y,cost_matrix_array,reg=reg,method='sinkhorn_stabilized',numItermax=1000)
        coupling = coupling/coupling.sum()
    else:
        return NotImplementedError
    distance = np.sum( coupling * cost_matrix_array )
    return h_x, h_y, bins_x_t.squeeze(-1),bins_y_t.squeeze(-1),torch.tensor(coupling), distance


# build categorical distribution associated to coupling
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

        # print(len(self.weights),len(self.samples),len(self.bins_x))
        # watch out for empty weights which can arise
        mask = torch.clone(self.weights)>1e-10
        self.weights = self.weights[mask]
        self.samples = self.samples[mask]

        # these have turned out to be too slow, let's just use a big catgorical
        self.categorical_distribution =D.Categorical(self.weights)

    def sample(self, n_samples):
        idx_sample = self.categorical_distribution.sample((n_samples,))
        return self.samples[idx_sample]




class FMModel:
    def __init__(self,
                 flow_network,
                 flow_network_opt,
                 flow_network_scheduler,
                 score_network,
                 score_network_opt, score_network_scheduler,
                 rho_0_dist,
                 device=torch.device('cuda:0')):

        self.flow_network = flow_network
        self.flow_network_opt = flow_network_opt
        self.flow_network_scheduler = flow_network_scheduler

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
        epsilon = torch.randn_like(x1)
        x = epsilon * sigma + mu

        # learn the flow
        flow_loss = nn.MSELoss()((x1 - x0),
                            self.flow_network(x, ts))
        # learn the score
        score_loss = nn.MSELoss()(- epsilon / sigma,
                            self.score_network(x, ts))

        return flow_loss, score_loss

    # sample ODE
    def sample(self, n_samples, n_time_steps=100):

        """
        Arguments:
            n_samples: int
            n_time_steps: int
        Returns:
            all_x_t: torch.Tensor of shape (n_samples, n_time_steps, n_dim)
        """

        # all_x_t = []
        # x_t = self.rho_0_dist.sample(torch.Size([n_samples,1]))
        # all_t = torch.linspace(0, 1, n_time_steps, device=self.device)
        # all_dt = torch.diff(all_t)
        # all_x_t.append(x_t)
        # for (dt, t) in zip(all_dt, all_t[:-1]):
        #     drift = self.flow_network(x_t, torch.ones(
        #         x_t.shape[0], 1, device=self.device).float() * t)
        #     x_t = x_t + drift * dt
        #     all_x_t.append(x_t)

        # all_x_t = torch.stack(all_x_t, dim=1)
        with torch.no_grad():
            x0 = self.rho_0_dist.sample(torch.Size([n_samples,1]))
            ts,all_xt, _ = langevin(x0,lambda t,xt : self.flow_network(xt,t),lambda t,xt : 0,nb_time_steps=n_time_steps)
        return ts, all_xt

    # recast ODE into SDE using the learned score
    def sample_diffusion(self, n_samples, n_time_steps=100, temperature=0.1):
        """
        Arguments:
            n_samples: int
            n_time_steps: int
        Returns:
            all_t: torch.Tensor of shape (n_samples, n_time_steps)
            all_xt: torch.Tensor of shape (n_samples, n_dim, n_time_steps)
            all_qt: torch.Tensor of shape (n_samples, n_time_steps)
        """
        # all_x_t = []
        # x_t = self.rho_0_dist.sample(torch.Size([n_samples,1]))
        # all_t = torch.linspace(0, 1, n_time_steps, device=self.device)
        # all_dt = torch.diff(all_t)
        # all_x_t.append(x_t)
        # for (dt, t) in zip(all_dt, all_t[:-1]):
        #     ts  = torch.ones(x_t.shape[0], 1, device=self.device).float() * t
        #     drift = self.flow_network(x_t,ts) + temperature * self.score_network(x_t,ts)
        #     x_t = x_t + drift * dt + torch.sqrt( 2 * temperature * dt ) * torch.randn_like(x_t)
        #     all_x_t.append(x_t)

        # all_x_t = torch.stack(all_x_t, dim=1)
        # return all_t,all_x_t
        with torch.no_grad():
                x0 = self.rho_0_dist.sample(torch.Size([n_samples,1]))
                ts,all_xt,all_qt = langevin(x0,lambda t,xt : self.flow_network(xt,t) + temperature *self.score_network(xt,t),lambda t,xt : temperature, nb_time_steps=n_time_steps)
        return ts, all_xt, all_qt

    def save(self, epoch_num=None):
        pass

def train_fm(fm_model, fm_dataloader, num_epochs=1000,
             save_epoch_freq=25,
             folder_name="./", tag="logs"):
    writer = SummaryWriter(log_dir=folder_name + tag)
    num_batches = len(fm_dataloader)
    for epoch in range(num_epochs):
        epoch_flow_loss = 0
        epoch_score_loss = 0
        for (train_x0, train_x1) in fm_dataloader:
            flow_loss, score_loss = fm_model.get_flow_matching_loss(train_x0, train_x1)

            fm_model.flow_network_opt.zero_grad()
            fm_model.score_network_opt.zero_grad()


            flow_loss.backward()
            score_loss.backward()

            fm_model.flow_network_opt.step()
            fm_model.score_network_opt.step()

            epoch_flow_loss += flow_loss.item()
            epoch_score_loss += score_loss.item()

        fm_model.flow_network_scheduler.step(epoch_flow_loss/num_batches)
        fm_model.score_network_scheduler.step(epoch_score_loss/num_batches)

        writer.add_scalar("Flow Matching Loss", epoch_flow_loss/num_batches, epoch)
        writer.add_scalar("Score Matching Loss", epoch_score_loss/num_batches, epoch)
        if epoch % save_epoch_freq == 0:
            fm_model.save(epoch_num=epoch)
