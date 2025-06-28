import numpy as np
import torch
epsilon = 1e-8

class NormalizedForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, normalize=True, data_mean=None, data_std=None):
        super().__init__()
        self.normalize = normalize
        print('normalize: ', self.normalize)

        if self.normalize:
            if data_mean is None or data_std is None:
                print('Calculating mean and std for normalization...')
                data_mean = np.mean(data, axis=0)
                data_std = np.std(data, axis=0)
                data_std = np.where(data_std == 0, epsilon, data_std)
                self.data_mean = data_mean
                self.data_std = data_std

            self.data = (data - data_mean) / data_std
        else:
            self.data = data

        self.window_size = window_size

        if data.shape[1] == 1:
            exit('error')
            data = data.squeeze()
            self.len, = data.shape
            self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, pred_len))

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i: i + self.window_size])
                Y[i, :] = torch.from_numpy(np.array(
                    data[i + self.window_size: i + self.window_size + pred_len]
                ))

            self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(Y, -1)

        else:
            self.len = self.data.shape[0]
            self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)

            X = torch.zeros((self.sample_num, self.window_size, self.data.shape[1]))
            Y = torch.zeros((self.sample_num, pred_len, self.data.shape[1]))

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(self.data[i: i + self.window_size, :])
                Y[i, :] = torch.from_numpy(self.data[i + self.window_size: i + self.window_size + pred_len, :])

            self.samples, self.targets = X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index, :, :], self.targets[index, :, :]
