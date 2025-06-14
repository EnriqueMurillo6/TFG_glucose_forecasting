from utils import interpolar_por_grupo
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

NUM_CLIENTS = 20
BATCH_SIZE = 32

def load_datasets(df, partition_id: int, max_encoder_length=12, max_prediction_length=6, min_encoder_length=6):
    """
    Carga datos para un cliente específico (partition_id) y devuelve DataLoaders
    preparados para Temporal Fusion Transformer (TFT).

    Args:
        df: dataframe completo con datos de todos los clientes/pacientes.
        partition_id: entero identificando la partición/cliente a cargar.
        max_encoder_length: longitud máxima del encoder.
        max_prediction_length: longitud máxima a predecir.
        min_encoder_length: longitud mínima del encoder (opcional).
    
    Returns:
        train_dataloader, val_dataloader
    """
    # Filtra cliente/partición
    df_client = df[df['archivo_id'] == partition_id].copy()

    # Interpolación y limpieza de valores faltantes
    df_client = interpolar_por_grupo(df_client, method='quadratic')
    df_client["carb"] = df_client["carb"].fillna(0)
    df_client["bolus"] = df_client["bolus"].fillna(0)
    df_client["basal_rate"] = df_client["basal_rate"].ffill().bfill()

    # Crear índice temporal por grupo
    df_client["time_idx"] = df_client.groupby("group_id").cumcount()
    df_client["group_id"] = df_client["group_id"].astype(str)

    # División train/val por índice temporal
    cutoff = df_client['time_idx'].max() - max_prediction_length
    df_train = df_client[df_client['time_idx'] <= cutoff]
    df_val = df_client[df_client['time_idx'] > cutoff]

    # Crear datasets TFT
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="Value",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=min_encoder_length if min_encoder_length else max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=["time_idx", "bolus", "basal_rate", "carb"],
        time_varying_unknown_reals=["Value"],
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)

    # DataLoaders
    train_dataloader = DataLoader(training, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader
