import torch
import torch.utils.data
import numpy as np
epsilon = 1e-8


class ReconstructNormalizedDataset(torch.utils.data.Dataset):

    def __init__(self, data, window_size, stride=1, normalize=True, data_mean=None, data_std=None):
        super().__init__()
        self.normalize = normalize

        if self.normalize:
            print('Normalizing data...')
            if data_mean is None or data_std is None:
                data_mean = np.mean(data, axis=0)
                data_std = np.std(data, axis=0)
                data_std = np.where(data_std == 0, epsilon, data_std)
            self.data_mean = data_mean
            self.data_std = data_std

            self.data = (data - data_mean) / data_std
            data = self.data
        else:
            self.data = data

        self.window_size = window_size
        self.stride = stride

        if data.shape[1] == 1:
            data = data.squeeze()
            self.len, = data.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.stride + 1)

            X = torch.zeros((self.sample_num, self.window_size))

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i * stride: i * stride + self.window_size])

            self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(X, -1)
        else:
            self.sample_num = data.shape[0]

            X = torch.zeros((data.shape[0], data.shape[1]))
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i, :])

            self.samples, self.targets = X, X

    def __len__(self):
        if self.data.shape[1] == 1:
            return self.sample_num
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        if self.data.shape[1] == 1:
            return self.samples[index, :, :], self.targets[index, :, :]
        else:
            if index < self.data.shape[0] - self.window_size:
                return self.samples[index:index + self.window_size, :], self.targets[index:index + self.window_size, :]
            else:
                return self.samples[-self.window_size:, :], self.targets[-self.window_size:, :]
