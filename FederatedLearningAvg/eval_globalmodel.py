import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data_patient_test(archivo_id
