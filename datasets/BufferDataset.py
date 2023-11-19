from torch.utils.data import Dataset


class BufferDataset(Dataset):
    def __init__(self, buffer, f0_min, f0_max, energy_min, energy_max):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.energy_min = energy_min
        self.energy_max = energy_max

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
