import argparse
import logging

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import SlidingWindowDataset
from xlstmad import xLSTMAD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='xLSTMAD example')
    parser.add_argument('--slstmbackend', type=str, default='cuda', help='slstm backend to use ("cuda" or "vanilla")')
    args = parser.parse_args()

    if args.slstmbackend not in ['cuda', 'vanilla']:
        raise ValueError('Invalid sltsm backend specified. Use "cuda" or "vanilla".')

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()], force=True)
    logger = logging.getLogger(__name__)

    batch_size = 32
    window_size = 20
    validation_size = 0.2

    # PREPARE DATA
    full_data = pd.read_csv('data/sample_dataset.csv').values
    train_length = 4000
    train_val_split = int(train_length * (1 - validation_size))
    train_data = full_data[:train_val_split]
    val_data = full_data[train_val_split:train_length]
    test_data = full_data[train_length:]
    train_x, train_y = train_data[:, :-1], train_data[:, -1]
    val_x, val_y = val_data[:, :-1], val_data[:, -1]
    test_x, test_y = test_data[:, :-1], test_data[:, -1]

    logging.info(f'train data size:{len(train_data)}')
    logging.info(f'validation data size:{len(val_data)}')
    logging.info(f'test data size:{len(test_data)}')

    train_loader = DataLoader(
        SlidingWindowDataset(train_x, train_y, window_size=window_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        SlidingWindowDataset(val_x, val_y, window_size=window_size),
        batch_size=4 * batch_size,
        shuffle=False,
        num_workers=4
    )

    # PREPARE MODEL
    try:
        model = xLSTMAD(embedding_dim=40, features_no=train_x.shape[1], window_size=window_size, slstm_backend=args.slstmbackend)
    except RuntimeError as e:
        if "Error building extension 'slstm" in str(e):
            logger.error("Failed to build the slstm extension. "
                         "The most common reason for this error is not having CUDA computing capability >8.0 or not having a compatible CUDA toolkit installed. "
                         "You can see more docs at xLSTM repository: https://github.com/NX-AI/xlstm"
                         "You can try to use the script with --slstmbackend=vanilla")
        logger.error(f"Original error: {e}")
        raise RuntimeError("Failed to initialize the model. See previous logs for details.") from e

    # PREPARE TRAINER
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min")

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="gpu",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=1e-4),
            checkpoint_cb],
        logger=True,
        enable_progress_bar=True,
        limit_train_batches=5,
        limit_test_batches=5
    )

    print(f'Trainer log file {trainer.log_dir}')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    print(f'Loading best model from {checkpoint_cb.best_model_path}')
    model = model.__class__.load_from_checkpoint(checkpoint_cb.best_model_path)

    test_loader = DataLoader(
        SlidingWindowDataset(test_x, test_y, window_size=window_size),
        batch_size=4 * batch_size,
        shuffle=False,
        num_workers=4
    )

    res = trainer.test(model, dataloaders=test_loader)
    logging.info(res)
