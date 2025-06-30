import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

class CustomTFT(TemporalFusionTransformer):
    def __init__(self, hidden_size, output_size=1, cnn_out_channels=64, cnn_kernel_size=3, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_>            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=hidden_size, kernel_size=1),
        )

        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x_out = super().forward(x)

        if isinstance(x_out, dict) and "prediction" in x_out:
            x_out = x_out["prediction"]

        if isinstance(x_out, torch.Tensor):
            x_cnn = x_out.permute(0, 2, 1)  
            x_cnn = self.cnn(x_cnn)
            x_out = x_cnn.permute(0, 2, 1) 
            x_out = x_out.mean(dim=1) 
            x_out = self.output_layer(x_out)  
        return x_out

def load_model(learning_rate, training):
    model = CustomTFT.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=512,
        attention_head_size=2,
        lstm_layers=2,
        dropout=0.2,
        output_size=1,
        loss=QuantileLoss(quantiles=[0.5]),
        log_interval=10,
    )
    return model
