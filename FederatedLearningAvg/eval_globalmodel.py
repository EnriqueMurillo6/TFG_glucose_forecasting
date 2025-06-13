import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import MAX_PREDICTION_LENGTH, MAX_ENCODER_LENGTH, MIN_ENCODER_LENGTH

def load_data_patient_test(df, archivo_id):
  df_test = df[df["archivo_id"] == archivo_id].copy()
  df_test["time_idx"] = df_test.groupby("group_id").cumcount()
  df_test["group_id"] = df_test["group_id"].astype(str)
    
  training_cutoff = df_test["time_idx"].max() - MAX_PREDICTION_LENGTH
    
  # Crear dataset específico para el paciente
  dataset_test = TimeSeriesDataSet(
      df_test[df_test.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="Value",
      group_ids=["group_id"],
      max_encoder_length=MAX_ENCODER_LENGTH,
      min_encoder_length=MIN_ENCODER_LENGTH,
      max_prediction_length=MAX_PREDICTION_LENGTH,
      time_varying_known_reals=["time_idx", "bolus", "basal_rate", "carb"],
      time_varying_unknown_reals=["Value"],
      target_normalizer=EncoderNormalizer(),
      add_relative_time_idx=True,
      add_target_scales=True,
      add_encoder_length=True,
    )
    
    return dataset_test

def load_model(path_model, dataset_test
