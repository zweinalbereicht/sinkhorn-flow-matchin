import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# generally speaking this  network will parametrize a force with a certain number of parameters
class ForceNetwork(nn.Module):
    """
    general network with time dependent forces parametrized by nb_parmas parameters
    """
    def __init__(self,nb_nodes,nb_params,force):
        super().__init__()
        self.relu_stack1=nn.Sequential(
            nn.Linear(1, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),nn.Linear(nb_nodes, nb_params)
        )
        self.force=force

    def forward(self,t):
        return self.relu_stack1(t) #(nbatch,ndim)

    def apply_force(self, positions,times): #eturns a (nbatch,2) force vector
        """
        Arguments:
            positions: torch.Tensor of shape (batch_size, 2)
            times: torch.Tensor of shape (batch_size, 1)
        """
        params = torch.vmap(self.forward)(times) #(nbatch,nparams)
        forces = self.force(positions,params) #(nbatch,2)
        return forces

class ForceModel:
    def __init__(self,
                 flow_network,
                 flow_network_opt,
                 flow_network_scheduler,
                 device=torch.device('cpu')):

        self.flow_network = flow_network
        self.flow_network_opt = flow_network_opt
        self.flow_network_scheduler = flow_network_scheduler
        self.device = device

    def get_loss(self, x0, x1):
        """
        Arguments:
            x0: torch.Tensor of shape (batch_size, 3)
            x1: torch.Tensor of shape (batch_size, 2)
            sigma: float
        """
        xt,t = torch.split(x0,(2,1),dim=1)
        # loss = nn.MSELoss()(x1, self.flow_network.apply_force(xt,t))

        loss = torch.mean(torch.sum((x1 - self.flow_network.apply_force(xt,t)) ** 2 * torch.max(t) / t, dim=1))
        return loss


class TestNetwork(nn.Module):
    def __init__(self, nb_nodes,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu_stack = nn.Sequential(
            nn.Linear(3, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, 2))

    def forward(self, x):
        return self.relu_stack(x)

class TestModel:
    def __init__(self,
                 flow_network,
                 flow_network_opt,
                 flow_network_scheduler,
                 device=torch.device('cpu')):

        self.flow_network = flow_network
        self.flow_network_opt = flow_network_opt
        self.flow_network_scheduler = flow_network_scheduler
        self.device = device

    def get_loss(self, x0, x1):
        """
        Arguments:
            x0: torch.Tensor of shape (batch_size, 3)
            x1: torch.Tensor of shape (batch_size, 2)
            sigma: float
        """
        loss = nn.MSELoss()(x1, self.flow_network(x0))
        return loss




def train_fm(fm_model, fm_dataloader, num_epochs=1000,
             save_epoch_freq=25,
             folder_name="./", tag="logs"):
    writer = SummaryWriter(log_dir=folder_name + tag)
    num_batches = len(fm_dataloader)
    for epoch in range(num_epochs):
        epoch_flow_loss = 0
        for (train_x0, train_x1) in fm_dataloader:
            # print(train_x0.shape,train_x1.shape)
            loss = fm_model.get_loss(train_x0, train_x1)

            fm_model.flow_network_opt.zero_grad()

            loss.backward()

            fm_model.flow_network_opt.step()

            epoch_flow_loss += loss.item()

        fm_model.flow_network_scheduler.step(epoch_flow_loss/num_batches)

        writer.add_scalar("Loss", epoch_flow_loss/num_batches, epoch)