import torch
import numpy as np


# A bunch of useful functions related to optimal transport

# taking fractional exponent values of a positive definite matrix
def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eigh(matrix)
    vals_pow = vals.pow(p)
    print(vals)
    # vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow

def wassertein_distance_gaussian(m1,cov1,m2,cov2):
    """
    returns the squared wasersetin distance between two (not necessarily 1d )gaussians with means and covs m1,cov1,m2,cov2
    """
    cov1_sqrt = _matrix_pow(cov1,0.5)
    bures_matrix = cov1 + cov2 - 2 * torch.sqrt(torch.matmul(cov1_sqrt, torch.matmul(cov2, cov1_sqrt)))
    return torch.sum((m1-m2) ** 2)+torch.trace(bures_matrix)

#  For now only works in the 1D/isotropic case
def wassertsein_interpolant_intermediate_gaussian(m1,cov1,m2,cov2,T,t):
    """
    input :
        m1,cov1,m2,cov2 : means and covs of the initial and final gaussians
        T : final time
        t : intermediate time
    returns :
        mean and cov of the intermediate gaussian
    """
    mut = (1 - t / T) * m1 + t / T * m2
    vt = (1 - t / T) ** 2 * cov1 + ( t / T ) ** 2 * cov2 + 2 * (1 - t / T) * (t / T) * np.sqrt(cov1 * cov2)
    return mut,vt

def KL_gaussian(m1,cov1,m2,cov2):
    """
    returns the kl divergence D(G1 || G2 ) between two (not necessarily 1d ) gaussians with means and covs m1,cov1,m2,cov2
    """

    det1 = torch.det(cov1)
    det2 = torch.det(cov2)
    inv2  = torch.inverse(cov2)
    trace = torch.trace(torch.matmul(inv2,cov1))

    return 1 / 2 * (torch.log(det2 / det1) - m1.shape[0] + trace + torch.matmul((m2-m1).T,torch.matmul(inv2,m2-m1)))


def langevin(x0 : torch.tensor, force, diffusion_matrix, final_time = 1.0, nb_time_steps=100):
    """
    Runs the Langevin Dynamics dXt = force(t,Xt)dt + sqrt(2 D(t, Xt))dWt between t=0 and t=1
    Args:
        x0 : initial points (nb_samples, nb_dim)
        force : function of time and space
        diffusion_matrix : function of time and space
        nb_time_steps : number of time steps

    Returns:
        ts : time steps
        all_xt : all the Xt (nb_samples,nb_time_steps,nbdim)
        all_qt : all the startonovitch heat increments (nb_samples, nb_time_steps)
    """
    ts = torch.linspace(0,final_time,nb_time_steps, device = 'cpu').float()
    dts = torch.diff(ts)
    nb_samples = x0.shape[0]
    xt = x0.clone()

    all_xt = []
    all_qt = []
    all_xt.append(xt.unsqueeze(1))

    k=1
    for (t,dt) in zip(ts[1:],dts):
        prev_xt = xt.clone()
        time = torch.ones((xt.shape[0],1)) * t
        drift = force(time,xt) * dt
        # we hardcode the two here
        difusion = torch.sqrt( 2 * diffusion_matrix(time,xt) * torch.abs(dt)) * torch.randn_like(xt)
        xt = xt + drift + difusion
        all_xt.append(xt.unsqueeze(1))


        # compute stratonovitch heat increment
        xt_center = (prev_xt + xt) / 2
        dxt = xt - prev_xt
        qt = torch.sum(dxt * force(time,xt_center),dim=1)


        all_qt.append(qt.unsqueeze(-1))
        k+=1
    return ts,torch.cat(all_xt,dim=1),torch.cat(all_qt,dim=-1)


def field_dicrepancy(final_time,times, data, field_1, field_2) :
    """
    Computes the field discrepancy between two fields, given some data
    Args:
        times : times of the data (nb_samples, nb_time_steps, nb_dim)
        data : (nb_samples, nb_time_steps, nb_dim)
        field_1 : function of time and space taking input of the form (xt,t)
        field_2 : function of time and space taking input of the form (xt,t)
    Returns:
        discrepancy : number
    """

    all_t = times.unsqueeze(0).repeat(data.shape[0],1).unsqueeze(-1)
    c = torch.cat((data,all_t),dim=-1)
    c= c.reshape(c.shape[0] * c.shape[1],c.shape[2])
    # print(c.shape)
    f1 = torch.vmap(field_1)(c)
    f2 = torch.vmap(field_2)(c)



    return final_time * torch.mean(torch.sum((f1 - f2)**2,dim=1))

def lower_bound(was, discrepency, beta, kappa, final_time) :
    """
    Computes the lower bound of the log likelihood
    Args:
        was : wasersetin distance not squared !!
        discrepency : discrepancy
        beta : float
        kappa : float
    Returns:
        lower_bound : float
    """
    return 1/ final_time * (was - np.sqrt(beta / 2 * discrepency / kappa)) ** 2













if __name__ == "__main__":
    m1 = torch.tensor([0.0])
    m2 = torch.ones(1) * 10.0
    cov1 = torch.eye(1)
    cov2 = torch.eye(1) * 1
    print(wassertein_distance_gaussian(m1,cov1,m2,cov2))
    print('nono')
