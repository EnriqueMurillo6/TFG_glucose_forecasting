from typing import List, Tuple
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
import os
import pickle
from flwr.common import parameters_to_ndarrays
from logging import INFO
from flwr.common.logger import log
import numpy as np

def save_global_model(weights, filename="global_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(weights, f)

NUM_ROUNDS=15

class CustomFedAvg(FedAvg):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.best_rmse_so_far = float("inf")

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None and server_round == self.num_rounds:
            weights = aggregated[0]
            save_global_model(weights, f"global_model_silos.pkl")

        return aggregated

def server_fn(context: Context):
    strategy =CustomFedAvg(
        num_rounds=NUM_ROUNDS,
        fraction_evaluate=1,
        min_fit_clients=4, # 4 silos de entrenamiento, usaremos otro silo para evaluación del modelo global.
        min_available_clients=4,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)

