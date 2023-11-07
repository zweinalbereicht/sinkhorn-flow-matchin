
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

def train_fm(fm_model, fm_dataloader, num_epochs=1000,
             save_epoch_freq=25,
             folder_name="./", tag="logs"):
    writer = SummaryWriter(log_dir=folder_name + tag)
    num_batches = len(fm_dataloader)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for (train_x0, train_x1) in fm_dataloader:
            loss = fm_model.get_flow_matching_loss(train_x0, train_x1)
            fm_model.score_network_opt.zero_grad()
            loss.backward()
            fm_model.score_network_opt.step()

            epoch_loss += loss.item()

        fm_model.score_network_scheduler.step(epoch_loss/num_batches)
        writer.add_scalar("Flow Matching Loss", epoch_loss/num_batches, epoch)
        if epoch % save_epoch_freq == 0:
            fm_model.save(epoch_num=epoch)
