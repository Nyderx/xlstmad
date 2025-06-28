import numpy as np
import torch
import torchinfo
import tqdm
from TSB_AD.utils.dataset import ForecastDataset
from TSB_AD.utils.torch_utility import get_gpu, EarlyStoppingTorch
from torch import nn, optim
from torch.utils.data import DataLoader
from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, \
    FeedForwardConfig, xLSTMBlockStack




class RandomModel():
    def __init__(self,
                 window_size=100,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 hidden_dim=40,
                 num_layer=2,
                 validation_size=0.2, ):
        super().__init__()
        self.validation_size = validation_size
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size


    def fit(self, data):
        pass

    def decision_function(self, data):
        anomaly_scores = np.random.rand(data.shape[0])
        self.__anomaly_score = anomaly_scores
        return anomaly_scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def get_y_hat(self) -> np.ndarray:
        return []

    def param_statistic(self, save_file):
        pass
