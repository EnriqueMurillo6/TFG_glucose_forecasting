from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

NUM_CLIENTS = 4 
BATCH_SIZE = 512

def load_datasets(df_silo, max_encoder_length=18, max_prediction_length=12):
    df_silo["group_id"] = df_silo["group_id"].astype(str)

    cutoff = df_silo['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df_silo[df_silo.time_idx <= cutoff],
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

    validation = TimeSeriesDataSet.from_dataset(training, df_silo, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(training, batch_size=BATCH_SIZE)
    val_dataloader = validation.to_dataloader(validation, batch_size=BATCH_SIZE)

    return train_dataloader, val_dataloader, training
