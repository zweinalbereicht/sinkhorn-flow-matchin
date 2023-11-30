import torch
import numpy as np
import matplotlib.pyplot as plt

def langevin(x0 : torch.tensor, force, diffusion_matrix, nb_time_steps=100):
    ts = torch.linspace(0,1,nb_time_steps, device = 'cpu').float()
    dts = torch.diff(ts)
    nb_samples = x0.shape[0]
    xt = x0.clone()
    all_xt = torch.zeros((nb_samples,nb_time_steps))
    all_xt[:,0] = xt.squeeze()
    k=1
    for (t,dt) in zip(ts[1:],dts):
        time = torch.ones_like(xt) * t
        drift = force(time,xt) * dt
        difusion = torch.sqrt( diffusion_matrix(time,xt) * torch.abs(dt)) * torch.randn_like(xt)
        xt = xt + drift + difusion
        all_xt[:,k]=xt.squeeze()
    return xt


if __name__ == '__main__':
    print('test')
    x0 = torch.randn(10,1)
    traj = langevin(x0, lambda t,xt : xt, lambda t,xt : torch.ones_like(xt), nb_time_steps=100)
    plt.plot(traj[0])
    plt.show()