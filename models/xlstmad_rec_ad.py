import numpy as np
import torch
import torchinfo
import tqdm
from TSB_AD.utils.dataset import ForecastDataset, ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu, EarlyStoppingTorch
from torch import nn, optim
from torch.utils.data import DataLoader
from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, \
    FeedForwardConfig, xLSTMBlockStack

from models.reconstruct_dataset import ReconstructNormalizedDataset


def create_config(window_size, features, embedding_dim=55):
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=8, qkv_proj_blocksize=5, num_heads=4, round_proj_up_dim_up=False,
                round_proj_up_to_multiple_of=5, embedding_dim=embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu", embedding_dim=embedding_dim),
        ),
        context_length=window_size,
        num_blocks=3,
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )


class xLSTMModel(nn.Module):
    def __init__(self, window_size, feats,
                 lstm_embedding_dim, batch_size, device) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.feats = feats
        self.device = device
        self.window_size = window_size

        self.encoder_projection = nn.Linear(feats, lstm_embedding_dim)

        cfg = create_config(window_size=window_size, features=feats, embedding_dim=lstm_embedding_dim)
        self.lstm_encoder = xLSTMBlockStack(cfg)
        self.lstm_decoder = xLSTMBlockStack(cfg)

        self.relu = nn.GELU()
        self.fc = nn.Linear(lstm_embedding_dim, feats)

    def forward(self, src):
        projected_input = self.encoder_projection(src)
        encoder_output = self.lstm_encoder(projected_input)
        decoder_hidden = encoder_output[:, -1]
        decoder_hidden = decoder_hidden.reshape(decoder_hidden.shape[0], 1, decoder_hidden.shape[1])
        cur_batch = src.shape[0]

        outputs = torch.zeros(self.window_size, cur_batch, self.feats).to(self.device)

        for t in range(self.window_size):
            decoder_hidden = self.lstm_decoder(decoder_hidden)
            decoder_output = self.relu(decoder_hidden)
            decoder_input = self.fc(decoder_output)

            outputs[t] = torch.squeeze(decoder_input, dim=-2)

        return outputs


class XLSTMADRec():
    def __init__(self,
                 window_size=100,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 hidden_dim=20,
                 validation_size=0.2, ):
        super().__init__()
        self.__anomaly_score = None

        cuda = True
        self.y_hats = None

        self.cuda = cuda
        self.device = get_gpu(self.cuda)

        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.feats = feats
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.validation_size = validation_size

        print('self.device: ', self.device)
        print(f'Window Size: {self.window_size}, ')

        self.model = xLSTMModel(self.window_size, feats, lstm_embedding_dim=self.hidden_dim,
                                batch_size=self.batch_size, device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)

        self.mu = None
        self.sigma = None
        self.eps = 1e-10

    def fit(self, data):
        tsTrain = data[:int((1 - self.validation_size) * len(data))]
        tsValid = data[int((1 - self.validation_size) * len(data)):]

        train_dataset = ReconstructNormalizedDataset(tsTrain, window_size=self.window_size)
        self.data_std = train_dataset.data_std
        self.data_mean = train_dataset.data_mean

        train_loader = DataLoader(
            ReconstructNormalizedDataset(tsTrain, window_size=self.window_size),
            batch_size=self.batch_size,
            shuffle=True)

        valid_loader = DataLoader(
            ReconstructNormalizedDataset(tsValid, window_size=self.window_size, data_std=train_dataset.data_std, data_mean=train_dataset.data_mean),
            batch_size=self.batch_size,
            shuffle=False)

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(x)

                output = output.view(-1, self.feats * self.window_size)
                target = target.view(-1, self.feats * self.window_size)

                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x)

                    output = output.view(-1, self.feats * self.window_size)
                    target = target.view(-1, self.feats * self.window_size)

                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())

            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs - 1:
                # fitting Gaussian Distribution
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                    print(self.mu.size(), self.sigma.size())
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            ReconstructNormalizedDataset(data, window_size=self.window_size, data_std=self.data_std, data_mean=self.data_mean),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)

                output = output.view(-1, self.feats * self.window_size)
                target = target.view(-1, self.feats * self.window_size)

                mse = torch.sub(output, target).pow(2)
                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)

        scores = scores.numpy()
        scores = np.mean(scores, axis=1)

        assert scores.ndim == 1
        if scores.shape[0] < len(data):
            exit('something wrong with the scores shape')
        #
        self.__anomaly_score = scores
        return scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def get_y_hat(self) -> np.ndarray:
        return self.y_hats

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
