import flwr
import pandas as pd
from client import FlowerClient
from data import cargar_datos_paciente

df_train = pd.read_parquet("df_train.parquet")
archivo_ids = df_train["archivo_id"].unique()

for archivo_id in archivo_ids:
    dataset_paciente = cargar_datos_paciente(df_train, archivo_id)
    modelo_paciente = CustomTFT.from_dataset(dataset_paciente, learning_rate=LEARNING_RATE)

    flwr.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient(modelo_paciente, dataset_paciente)
    )
