import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
torch.set_printoptions(sci_mode=False)


class Data():
    def __init__(self, device=torch.device("cpu"),data_source="filename",init_samples=torch.randn(1000), target_samples=torch.randn(1000)):
        self.device = device
        self.bins = np.arange(-10, 10, 0.5, dtype=np.float32) # hardcoding the bins might be an issue actually
        self.bin_size = torch.diff(torch.tensor(self.bins))[0]
        self.init_samples = init_samples
        self.target_samples = target_samples
        if data_source == "filename":
            self.target_dist, self.current_dist = self._init_data()
        else :
            self.bins, self.target_dist, self.current_dist = self._init_data_from_samples(self.init_samples,self.target_samples)
            self.bin_size = torch.diff(torch.tensor(self.bins))[0]

    def _init_data(self, filename = "./data/high_temp_clusters_shape.npy"):
        """Initializes target_dist and current_dist
        Arguments:
            filename: Name of numpy file of initial cluster sizes to be used to define initial distribution
        Returns:
            target_dist: A tensor of shape (num_bins_init, 1) representing the final distribution
            init_dist: A tensor of shape (num_bins_final, 1) representing the initial distribution
        """
        target_data = np.random.gamma(16, 0.25, 10000000)
        target_data = np.histogram(target_data, bins=self.bins, density=True)
        initial_data = np.load(filename)
        initial_data = np.histogram(initial_data, bins=self.bins, density=True)
        init_dist = torch.tensor(initial_data[0], device=self.device).float()
        init_dist += 1E-4 #Add elements in each bin (just to make sure we never divide by 0)
        init_dist /= torch.sum(init_dist)
        return (torch.tensor(target_data[0], device=self.device).float(),
                init_dist)

    def _init_data_from_samples(self, init_samples, target_samples):
        """Initializes target_dist and current_dist
        Arguments:
            *_samples: samples from the base and target distributions, not yet binned.
        Returns:
            target_dist: A tensor of shape (num_bins_init, 1) representing the target distribution
            init_samples: A tensor of shape (num_bins_final, 1) representing the initial distribution
        """
        leftmost_bin = torch.min(torch.min(init_samples),torch.min(target_samples))-2.0
        rightmost_bin = torch.max(torch.max(init_samples),torch.max(target_samples))+2.0
        bins = np.arange(int(leftmost_bin), int(rightmost_bin), 0.5, dtype=np.float32)
        target_data = np.histogram(target_samples, bins=bins, density=True)
        initial_data = np.histogram(init_samples, bins=bins, density=True)
        init_dist = torch.tensor(initial_data[0], device=self.device).float()

        reg = 1/len(init_samples)
        init_dist += reg #Add elements in each bin
        init_dist /= torch.sum(init_dist)
        target_dist = torch.tensor(target_data[0], device=self.device).float()
        target_dist += reg #Add elements in each bin
        target_dist /= torch.sum(target_dist)
        return (bins, target_dist,init_dist)


class Sinkhorn():
    def __init__(self, device=torch.device("cpu"), train=True, source="samples",init_samples=torch.randn(1000), target_samples=torch.randn(1000)):
        self.device = device
        self.data = Data(self.device,data_source=source,init_samples=init_samples, target_samples=target_samples)
        xs = np.array(self.data.bins[:-1])
        self.push_forward = None
        if (train):
            self._train()

    def sinkhorn(self, xs, ys, w_1=None, w_2=None, n_iter=1000, eps=0.01):
        """Applies sinkhorn algorithm
        Arguments:
            xs: A tensor of shape (num_bins_init, 1)
            ys: A tensor of shape (num_bins_final, 1)
            w_1: weights for xs, if None assumes equal weights (this is "a" in comptutaional optimal transport p64 --> it is the marginal distribution )
            w_2: weights for ys, if None assumes equal weights
            n_iter: Number of iterations to train
            eps: Entropic Regularization value
        Returns:
            A torch tensor of (num_bins_init, num_bins_final) representing the
            trained push forward --> dat's going to be trickier with higher dimensions...
        """

        def dist_mat(xs, ys):
            z = xs - ys.t()# returns a squared matrix of distances (useful trick)
            return torch.abs(z)
        with torch.no_grad():
            k = xs.shape[0]
            l = ys.shape[0]
            if w_1 is None:
                w_1 = torch.ones(k, 1, device=self.device) / k
            if w_2 is None:
                w_2 = torch.ones(l, 1, device=self.device) / l

        dij = dist_mat(xs, ys)
        # pretty sure we're gonne need a square in here --> might be important if we don"t put it
        K = torch.exp(-dij ** 2 / eps)
        u = torch.ones((k, 1), device=self.device, requires_grad=True)
        v = torch.ones((l, 1), device=self.device, requires_grad=True)

        for i in range(n_iter):
            u = w_1 / ((K @ v) + 1e-32)
            v = w_2 / ((K.T @ u) + 1e-32)

        #rounding step as described in Culturi at al
        up = u * torch.min(w_1 / (u * (K @ v)),torch.ones_like(u))
        vp = v * torch.min(w_2 / (v * (K.T @ up)),torch.ones_like(up))
        delta_a = w_1 - up * (K @ vp)
        delta_b = w_2 - vp * (K.T @ u)

        return torch.diag(up.squeeze(1)) @ K @ torch.diag(vp.squeeze(1)) +  ( delta_a @ delta_b.T ) / torch.norm(delta_a,1)

    def _train(self,eps=1,n_iter=1000):
        """Trains self.push_forward map based on sinkhorn algorithm
        """
        xs = torch.tensor(self.data.bins[:-1], device=self.device).unsqueeze(1) # arbitrary bin crop but okay
        ys = xs.clone()
        self.push_forward = self.sinkhorn(xs,
                                          ys,
                                          self.data.current_dist.unsqueeze(1),
                                          self.data.target_dist.unsqueeze(1),
                                          n_iter=n_iter,
                                          eps=eps)

        # print(torch.sum(self.push_forward, dim=1))
        # print(self.data.current_dist)
        assert(torch.allclose(torch.sum(self.push_forward, dim=1),
                              self.data.current_dist)) # checks if we have have the marginals right

        assert(torch.allclose(torch.sum(self.push_forward, dim=0),
                              self.data.target_dist))

    def get_q(self, t_frac):
        """Gets the intermediate probability distribution at t_frac along transport
        Arguments:
            t_frac: A float between 0 and 1 representing fraction of transport
        Returns:
            q: A torch tensor representing the intermediate  probability distribution
        """

        def get_bin(x, bs):
            """Check which bin x belongs to
            Arguments:
                x: Element
                bs: A torch tensor of all the bins
            Returns:
                idx: Index of the bin that x belongs to
            """
            db = bs[1] - bs[0]
            bin_edge = torch.where((x - bs).abs() < 1E-4)[0]
            if (bin_edge.numel()):
                """If x is on a bin_edge
                """
                return bin_edge
            return torch.nonzero((x > bs) * (x < bs + db), as_tuple=True)[0]
        bins = self.data.target_dist.size(0)
        xs = torch.arange(1, bins + 1, device=self.device)
        ys = torch.arange(1, bins + 1, device=self.device)
        #Constant speed geodesic to see how far probability mass has traveled
        yts = t_frac * ys.unsqueeze(0) + (1 - t_frac) * xs.unsqueeze(1)
        q = torch.zeros_like(ys, device=self.device).float()
        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                #Get the bin in which pmass has traveled and all the mass to that bin
                idx = get_bin(yts[i, j], ys)
                q[idx] += self.push_forward[i, j]

        return q


    def visualize(self, num_intervals=15, filename = "vis.gif"):
        """Visualizes transport and saves gif with name filename
        Arguments:
            num_intervals: Number of intervals along interval
            filename: filename of gif
        """
        xs = torch.tensor(self.data.bins[:-1], device=self.device)
        ys = torch.tensor(self.data.bins[:-1], device=self.device)
        fig=plt.figure()
        barcollection = plt.bar(xs, torch.randn(len(self.data.bins[:-1])).numpy())
        plt.ylim(0, 1)

        def update(t):
            x = t / num_intervals
            y = self.get_q(x)
            assert(torch.allclose(torch.sum(y),
                                  torch.tensor(1.0, device=self.device)))
            for i, b in enumerate(barcollection):
                b.set_height(y[i].item())

        ani = FuncAnimation(fig, update, frames=num_intervals + 1, repeat=True)
        writergif = PillowWriter(fps=5)
        ani.save("filename", writer=writergif)
