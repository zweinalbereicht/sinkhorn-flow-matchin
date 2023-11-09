
# structure to generate flow matching training and testing samples.
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class FMDataset(Dataset):
    def __init__(self, x0_samples, x1_samples):
        self.x0_samples = x0_samples
        self.x1_samples = x1_samples

    def __len__(self):
        return self.x0_samples.shape[0]

    def __getitem__(self, idx):
        return self.x0_samples[idx], self.x1_samples[idx]
