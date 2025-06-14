import torch
import numpy as np
from typing import List
from collections import OrderedDict

from flwr.client import NumPyClient, Client, ClientApp
from pytorch_forecasting.metrics import QuantileLoss
from data_loader import load_full_dataframe
from models import CustomTFT, get_parameters, set_parameters
from datasets import load_datasets
from train import train, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_full_dataframe("../df.parquet")


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)  # Usa tu función train adaptada
        # Si trainloader no tiene .dataset, puedes usar len(trainloader.dataset) o sum de batch sizes
        dataset_len = len(self.trainloader.dataset) if hasattr(self.trainloader, 'dataset') else sum(len(batch[0]['decoder_target']) for batch in self.trainloader)
        return get_parameters(self.net), dataset_len, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        self.net.eval()
        total_loss = 0.0
        preds = []
        targets = []

        with torch.no_grad():
            for batch in self.valloader:
                x = batch[0].to(DEVICE)  # inputs dict
                y = batch[0]['decoder_target'].to(DEVICE)  # target tensor

                output = self.net(x)
                if isinstance(output, dict) and "loss" in output and "prediction" in output:
                    loss = output["loss"]
                    prediction = output["prediction"]
                else:
                    raise RuntimeError("La salida del modelo no contiene 'loss' y 'prediction'.")

                total_loss += loss.item()
                preds.append(prediction.detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())

        avg_loss = total_loss / len(self.valloader)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        dataset_len = len(self.valloader.dataset) if hasattr(self.valloader, 'dataset') else sum(len(batch[0]['decoder_target']) for batch in self.valloader)

        return float(avg_loss), dataset_len, {"rmse": float(rmse)}


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]

    # Ojo con df, debe estar disponible aquí, o pásalo como argumento/importalo
    trainloader, valloader = load_datasets(df=df, partition_id=partition_id)

    training_dataset = trainloader.dataset

    net = CustomTFT.from_dataset(
        training_dataset,
        learning_rate=0.001,
        hidden_size=128,
        attention_head_size=2,
        lstm_layers=2,
        dropout=0.1,
        output_size=6,
        loss=QuantileLoss(quantiles=[0.5]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    ).to(DEVICE)

    return FlowerClient(net, trainloader, valloader).to_client()

client = ClientApp(client_fn=client_fn)
