import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def interpolar_por_grupo(df, method='quadratic'):
    dfs_interpolados = []

    for group_id, group in df.groupby('group_id'):
        group = group.sort_values('Zulu Time').reset_index(drop=True)

        diff = group['Zulu Time'].diff().shift(-1)
        gaps = diff[diff > pd.Timedelta(minutes=5)]

        new_index = group['Zulu Time'].tolist()
        for idx in gaps.index:
            start = group.loc[idx, 'Zulu Time']
            end = group.loc[idx + 1, 'Zulu Time']
            new_times = pd.date_range(start=start, end=end, freq='5min')
            new_index.extend(new_times[:-1])

        new_index = sorted(set(new_index))
        reindexed_group = group.set_index('Zulu Time').reindex(new_index).reset_index()

        reindexed_group['Value'] = reindexed_group['Value'].fillna(method='ffill')

        reindexed_group['Value'] = reindexed_group['Value'].interpolate(method=method, limit_direction='both')

        reindexed_group['group_id'] = group_id
        reindexed_group['archivo_id'] = group['archivo_id'].iloc[0]

        dfs_interpolados.append(reindexed_group)

    return pd.concat(dfs_interpolados).reset_index(drop=True)

def load_data_patient(df, archivo_id):
    df_paciente = df[df["archivo_id"] == archivo_id].copy()
    df_paciente = interpolar_por_grupo(df_paciente, method='quadratic')
  
    df_paciente["carb"] = df_paciente["carb"].fillna(0)
    df_paciente["bolus"] = df_paciente["bolus"].fillna(0)
    df_paciente["basal_rate"] = df_paciente["basal_rate"].ffill().bfill()
    df_paciente["time_idx"] = df_paciente.groupby("group_id").cumcount()
    df_paciente["group_id"] = df_paciente["group_id"].astype(str)
    
    training_cutoff = df_paciente["time_idx"].max() - max_prediction_length
    
    # Crear dataset específico para el paciente
    dataset_paciente = TimeSeriesDataSet(
        df_paciente[df_paciente.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Value",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "bolus", "basal_rate", "carb"],
        time_varying_unknown_reals=["Value"],
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    return dataset_paciente
