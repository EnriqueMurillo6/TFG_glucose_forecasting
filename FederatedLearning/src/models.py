import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List
from pytorch_forecasting import TemporalFusionTransformer

class CustomTFT(TemporalFusionTransformer):
    def __init__(self, hidden_size=256, output_size=6, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extra_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_out = super().forward(x)

        if isinstance(x_out, dict) and "prediction" in x_out:
            x_out = x_out["prediction"]

        if isinstance(x_out, torch.Tensor):
            x_out, _ = self.extra_lstm(x_out.unsqueeze(1))
            x_out = x_out.squeeze(1)
            x_out = self.output_layer(x_out)
        return x_out

def set_parameters(net, parameters: List[torch.Tensor]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[torch.Tensor]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
