from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

NUM_CLIENTS = 15
BATCH_SIZE = 512

def load_datasets(df, partition_id: int, max_encoder_length=18, max_prediction_length=12):
    pacientes = df["archivo_id"].unique()
    paciente_id = pacientes[partition_id]
    df_client = df[df["archivo_id"] == paciente_id].copy()
    df_client["group_id"] = df_client["group_id"].astype(str)

    cutoff = df_client['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df_client[df_client.time_idx <= cutoff],
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

    validation = TimeSeriesDataSet.from_dataset(training, df_client, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(training, batch_size=BATCH_SIZE)
    val_dataloader = validation.to_dataloader(validation, batch_size=BATCH_SIZE)

    return train_dataloader, val_dataloader, training
