import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, RMSE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet
import os
from pytorch_lightning import Trainer, LightningModule
import torch.nn as nn
from pytorch_forecasting.data.encoders import EncoderNormalizer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

path_train = "/content/drive/MyDrive/TFG/Train-Test/df_train.parquet"
path_test = "/content/drive/MyDrive/TFG/Train-Test/df_test_elim.parquet"

df_train = pd.read_parquet(path_train)
df_test = pd.read_parquet(path_test)
df_train["time_idx"] = df_train.groupby("group_id").cumcount()
df_train["group_id"] = df_train["group_id"].astype(str)

def clarke_error_grid(ref_values, pred_values):
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(
        len(ref_values), len(pred_values))

    if ref_values.max() > 400 or pred_values.max() > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)))
    if ref_values.min() < 0 or pred_values.min() < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)))


    # Clear plot
    plt.clf()
    plt.style.use("seaborn-v0_8")

    # Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=1)
    plt.title("Gradilla de error de Clarke")
    plt.xlabel("Concentración de referencia (mg/dl)")
    plt.ylabel("Concentración predicha (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400) / (400))

    # Plot zone lines
    plt.plot([0, 400], [0, 400], ':', c='black')  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175 / 3, 400 / 1.2], [70, 400], '-',
             c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values.iloc[i] <= 70 and pred_values.iloc[i] <= 70) or (
                pred_values.iloc[i] <= 1.2 * ref_values.iloc[i] and pred_values.iloc[i] >= 0.8 * ref_values.iloc[i]):
            zone[0] += 1  # Zone A

        elif (ref_values.iloc[i] >= 180 and pred_values.iloc[i] <= 70) or (
                ref_values.iloc[i] <= 70 and pred_values.iloc[i] >= 180):
            zone[4] += 1  # Zone E

        elif ((ref_values.iloc[i] >= 70 and ref_values.iloc[i] <= 290) and pred_values.iloc[i] >= ref_values.iloc[
            i] + 110) or ((ref_values.iloc[i] >= 130 and ref_values.iloc[i] <= 180) and (
                pred_values.iloc[i] <= (7 / 5) * ref_values.iloc[i] - 182)):
            zone[2] += 1  # Zone C
        elif (ref_values.iloc[i] >= 240 and (pred_values.iloc[i] >= 70 and pred_values.iloc[i] <= 180)) or (
                ref_values.iloc[i] <= 175 / 3 and pred_values.iloc[i] <= 180 and pred_values.iloc[i] >= 70) or (
                (ref_values.iloc[i] >= 175 / 3 and ref_values.iloc[i] <= 70) and pred_values.iloc[i] >= (6 / 5) *
                ref_values.iloc[i]):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return plt, zone

def compute_mard(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> float:
    # Aplanamos para tener todas las predicciones y valores reales en un solo vector
    y_pred = predictions_df.to_numpy().flatten()
    y_true = actuals_df.to_numpy().flatten()

    # Evitamos divisiones por cero
    nonzero_mask = y_true != 0
    y_pred = y_pred[nonzero_mask]
    y_true = y_true[nonzero_mask]

    mard = np.mean(np.abs(y_pred - y_true) / y_true) * 100
    return mard


# Evaluación del modelo global para horizonte de 60 minutos. Solo habría que cambiar parámetros para otro horizonte.
batch_size=2048
max_encoder_length = 18

max_prediction_length = 12

training_cutoff = df_train["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_train[df_train.time_idx <= training_cutoff],
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

best_tft = CustomTFT.load_from_checkpoint("tft_model_60min.ckpt")

df_test["group_id"] = df_test["group_id"].astype(str)
df_test["time_idx"] = df_test.groupby("group_id").cumcount()

test = TimeSeriesDataSet.from_dataset(training, df_test, predict=True, stop_randomization=True)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

