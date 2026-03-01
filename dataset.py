import torch
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, x, y, window_size=128, drop_last=True):
        assert len(x) == len(y), "Data and labels must have the same length"

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)
        self.window_size = window_size
        self.drop_last = drop_last

    def __len__(self):
        return len(self.x) - self.window_size + 1

    def __getitem__(self, idx):
        end_idx = idx + self.window_size
        if end_idx > len(self.x):
            raise IndexError("Index out of range")

        return self.x[idx:end_idx], self.y[end_idx-1]