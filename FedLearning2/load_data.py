from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

def crear_dataloaders_por_cliente(
    df_cliente,
    max_encoder_length=12,
    min_encoder_length=6,
    max_prediction_length=6,
    batch_size=2048,
    num_workers=4
):
    df_cliente = df_cliente.copy()
    df_cliente["group_id"] = df_cliente["group_id"].astype(str)
    df_cliente["time_idx"] = df_cliente.groupby("group_id").cumcount()

    # Dataset para entrenamiento
    dataset = TimeSeriesDataSet(
        df_cliente[df_cliente.time_idx <= df_cliente["time_idx"].max() - max_prediction_length],
        time_idx="time_idx",
        target="Value",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=["time_idx", "bolus", "basal_rate", "carb"],
        time_varying_unknown_reals=["Value"],
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Dataloader de entrenamiento
    train_loader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)

    # Dataset de validación
    val_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        df_cliente,
        predict=True,
        stop_randomization=True
    )
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
