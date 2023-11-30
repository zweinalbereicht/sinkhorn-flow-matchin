import torch

# implement a force that dirves towards a stationary distribution given by a quadrimlodal mixture of gaussians
def quadrimodal(xt,beta, mu, sigma):
    """
    input :
        xt : (nb_dim, 1)
        beta : inverse temperature
        mu : real number
        sigma : real number

    output :
        force : (nb_dim, 1)

    the target is a quadrant centered mixture of gaussians, each having variance sigma and centered at mu
    """

    th = mu * torch.tanh(mu * xt / sigma ** 2)
    return 1 / (beta * sigma ** 2 ) * (th - xt)

# the optimal forcing to drive a system initialy distributed as a simple gaussian towards the top right
# quadrant of the quadrimodal potential.
def optmal_forcing_quadrimodal(xt ,t ,T ,beta, mu0 ,sigma0, mu, sigma, mu_pot = None, sigma_pot = None):
    """
    input :
        xt : (nb_dim, 1)
        t : time
        T : final time
        beta : inverse temperature
        mu0 : intial mean
        sigma0 : initial std
        mu : real number
        sigma : real number

    output :
        force : (nb_dim, 1)

    the target is a the top right well of the quadrimodal potential and we want to reach it the time T
    """

    # counter the force
    if mu_pot is None:
        f1 = - quadrimodal(xt,beta, mu, sigma)
    else :
        f1 = - quadrimodal(xt,beta, mu_pot, sigma_pot)

    drift = 1 / T * 1 /  ( sigma0 - t / T * (sigma0 - sigma))  * ( sigma0 * (mu - xt) + sigma * (xt - mu0))
    score = 1 / T * 1 /  ( sigma0 + t / T * (sigma - sigma0)) ** 2  * ( t * (mu - mu0) + T * (mu0 - xt))
    return f1 + drift + 1 / beta * score


def fixed_truncated_harmonic_force(positions : torch.Tensor, params : torch.Tensor, fixed_params : torch.Tensor ) -> torch.Tensor:
    """
    input :
        positions : (nbatch, 2)
        fixed params : (nbatch, 4 * n ) [[cx,cy,cutoffx,cutoffy]] these are fixed
        params : (nbatch, 4 * n ) [[kxx,kxy, kyx, kyy]] these are leaned by the network

    returns :
        force : (nbatch, 2)

    This one is a fixez
    """
    x,y = torch.split(positions,1,dim=1) #(nbatch,1) for x and y

    # need to have everyone in the same dimension
    nb_fixed_params = fixed_params.shape[0]
    traps_fixed_params = fixed_params.repeat(x.shape[0],nb_fixed_params) #(nbatch,,nb_fixed_params))

    traps_fixed_params=torch.split(traps_fixed_params,4,dim=1) #(nbatch,4) for each trap
    traps_params=torch.split(params,4,dim=1) #(nbatch,2) for each trap

    grad_u_x=torch.zeros((len(x),1))
    grad_u_y=torch.zeros_like(grad_u_x)

    # this is a hardcoded scale that will need to be tuned later or integrated in the network
    sm_scale=10

    for t in traps_fixed_params:
        cx,cy,cutoffx,cutoffy=torch.split(t,1,dim=1) #(nbatch,2)

    for t in traps_params:
        kxx,kxy,kyx,kyy=torch.split(t,1,dim=1) #(nbatch,2)

        # introduce a sigmoid violent cut off function here but it's fine
        distance_to_center = torch.max(torch.abs(x-cx),torch.abs(y-cy)) #(nbatch,1)

        grad_u_x = grad_u_x +  ( kxx * (x-cx) + kxy * (y-cy))* torch.sigmoid((cutoffx-distance_to_center) * sm_scale)  #(nbatch,1)
        grad_u_y = grad_u_y +( kyy * (y-cy) + kyx * (x - cx) ) * torch.sigmoid((cutoffy-distance_to_center) * sm_scale)

    # directly return the force
    return torch.cat((-grad_u_x,-grad_u_y),dim=1) #(nbatch,2)
