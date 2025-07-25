import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer, LightningModule
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import copy
from pathlib import Path
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer, EncoderNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import RMSE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

class CustomTFT(TemporalFusionTransformer):
    def __init__(self, hidden_size, output_size=1, cnn_out_channels=64, cnn_kernel_size=3, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=hidden_size, kernel_size=1),
        )

        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x_out = super().forward(x)

        if isinstance(x_out, dict) and "prediction" in x_out:
            x_out = x_out["prediction"]

        if isinstance(x_out, torch.Tensor):
            # x_out: (batch, time, hidden_size)
            x_cnn = x_out.permute(0, 2, 1)  
            x_cnn = self.cnn(x_cnn)
            x_out = x_cnn.permute(0, 2, 1)  

            x_out = x_out.mean(dim=1)  

            x_out = self.output_layer(x_out)  

        return x_out


path_train = "df_train.parquet"
path_test = "df_test.parquet"

df_train = pd.read_parquet(path_train)
df_test = pd.read_parquet(path_test)


df_train["time_idx"] = df_train.groupby("group_id").cumcount()
df_train["group_id"] = df_train["group_id"].astype(str)

max_encoder_length = 18

max_prediction_length = 12 

training_cutoff = df_train["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_train[df_train.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Value",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Value", "bolus", "basal_rate", "carb"],
    target_normalizer=EncoderNormalizer(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

df_test["group_id"] = df_test["group_id"].astype(str)
df_test["time_idx"] = df_test.groupby("group_id").cumcount()

test = TimeSeriesDataSet.from_dataset(training, df_test, predict=True, stop_randomization=True)
batch_size = 2048
learning_rate = 0.001
hidden_size = 512
dropout = 0.2

validation = TimeSeriesDataSet.from_dataset(training, df_train, predict=True, stop_randomization=True)
from pytorch_lightning import seed_everything

seed_everything(42)

from sklearn.metrics import mean_absolute_error, mean_squared_error

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

tft = CustomTFT.from_dataset(
      training,
      learning_rate=learning_rate,
      hidden_size=hidden_size,
      attention_head_size=2,
      lstm_layers=2,
      dropout=dropout,
      output_size=1,
      loss=QuantileLoss(quantiles=[0.5]),
      log_interval=10,
      reduce_on_plateau_patience=4,
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback],
)

trainer.fit(tft, train_dataloader, val_dataloader)

checkpoint_path = trainer.checkpoint_callback.best_model_path
if checkpoint_path == "":
    print("No se encontró checkpoint para esta configuración")

model = CustomTFT.load_from_checkpoint(checkpoint_path)

predictions = model.predict(test_dataloader, mode="prediction")
predictions_df = pd.DataFrame(predictions.cpu().numpy())

actuals = []
for batch in iter(test_dataloader):
    targets = batch[1][0]
    actuals.append(targets)

actuals = torch.cat(actuals).numpy()
actuals_df = pd.DataFrame(actuals)
rmse = root_mean_squared_error(predictions_df, actuals_df)
print(f"RMSE test: {rmse:.4f}")

# trainer.save_checkpoint("tft_model_60min.ckpt")
