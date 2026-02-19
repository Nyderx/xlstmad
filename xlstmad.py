from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn, optim
from torch.nn import MSELoss
from torch.nn.functional import mse_loss
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAUROC
from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, \
    FeedForwardConfig, xLSTMBlockStack


def create_config(window_size, embedding_dim=55, backend="cuda"):
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=8, qkv_proj_blocksize=5, num_heads=4, round_proj_up_dim_up=False,
                round_proj_up_to_multiple_of=5, embedding_dim=embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=backend,
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu", embedding_dim=embedding_dim),
        ),
        context_length=window_size,
        num_blocks=5,
        embedding_dim=embedding_dim,
        slstm_at=[1, 2, 3],
    )


class xLSTMAD(L.LightningModule):
    def __init__(self, embedding_dim: int, features_no: int, window_size: int, lr: float = 0.001,
                 slstm_backend: str = "cuda"):
        super(xLSTMAD, self).__init__()
        self.window_size = window_size
        self.features_no = features_no
        self.lr = lr

        xlstm_cfg = create_config(window_size=window_size, embedding_dim=embedding_dim, backend=slstm_backend)
        self.encoder = xLSTMBlockStack(xlstm_cfg)
        self.decoder = xLSTMBlockStack(xlstm_cfg)

        self.input_projection = nn.Linear(features_no, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, features_no)
        self.gelu = nn.GELU()

        self.loss = MSELoss()
        self.val_loss = MSELoss()

        self.test_auc = BinaryAUROC()
        self.test_mse = MeanSquaredError()

        self.save_hyperparameters()

    def forward(self, x):
        projected_input = self.input_projection(x)
        encoder_output = self.encoder(projected_input)
        decoder_output = self.decoder(encoder_output)
        outputs = torch.zeros(self.window_size, x.shape[0], self.features_no).to(self.device)

        for step in range(self.window_size):
            output_activation = self.gelu(decoder_output[:, step, :])
            output_projected = self.output_projection(output_activation)
            outputs[step] = output_projected

        outputs = outputs.reshape(outputs.shape[1], outputs.shape[0], outputs.shape[2])
        return outputs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # x = x.permute(1, 0, 2)
        output = self.forward(x)
        loss = self.loss(output, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # x = x.permute(1, 0, 2)
        output = self.forward(x)
        loss = self.val_loss(output, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x_hat = self.forward(x)
        rec_error = mse_loss(x, x_hat, reduction='none').mean(dim=(1, 2))
        self.test_auc.update(rec_error, y.int())
        self.test_mse.update(x, x_hat)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True)
        self.log("test_mse", self.test_mse, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx) -> Any:
        x, target = batch
        reconstruction = self.forward(x)
        anomaly_scores = torch.mean(mse_loss(reconstruction, target, reduction='none'), dim=(1, 2))
        return anomaly_scores

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
