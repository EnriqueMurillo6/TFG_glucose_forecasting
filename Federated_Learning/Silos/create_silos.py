import pandas as pd
import os

df = pd.read_parquet("df_15.parquet")
assert "archivo_id" in df.columns, "La columna 'archivo_id' no está en el DataFrame."

unique_ids = df["archivo_id"].unique()
assert len(unique_ids) == 15, f"Se esperaban 15 pacientes, pero se encontraron {len(unique_ids)}"

# Dividir en 4 grupos (4, 4, 4, 3)
ranges = [(0, 4), (4, 8), (8, 12), (12, 15)]
silo_splits = [unique_ids[start:end] for start, end in ranges]

base_dir = "silos"
os.makedirs(base_dir, exist_ok=True)
