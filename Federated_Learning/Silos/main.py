import pandas as pd
from flwr.simulation import run_simulation
from server   import server, NUM_ROUNDS
from client import app
from datasets import NUM_CLIENTS

run_simulation(
    server_app=server,
    client_app=app,
    num_supernodes=NUM_CLIENTS,
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1}},
)
