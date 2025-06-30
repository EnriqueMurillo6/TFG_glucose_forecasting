import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn.model_selection import GroupShuffleSplit

df1 = pd.read_parquet('df_cgm.parquet')
df2 = pd.read_parquet('df_bolus.parquet')
df3 = pd.read_parquet('df_basal.parquet')
df4 = pd.read_parquet('df_food.parquet')

dfs = [df1, df2, df3, df4]
df = reduce(lambda left, right: pd.merge(
    left, right, on=['archivo_id', 'Zulu Time'], how='outer'), dfs)
df['Zulu Time'] = pd.to_datetime(df['Zulu Time'])
df = df.sort_values(['archivo_id', 'Zulu Time']).reset_index(drop=True)

df.rename(columns={'Normal': 'bolus', 'Rate': 'basal_rate'}, inplace=True)
df['carb'] = df['carb'].fillna(0)
df['bolus'] = df['bolus'].fillna(0)
df['basal_rate'] = df.groupby('archivo_id')['basal_rate'].ffill().bfill()

df['is_valid'] = df['Value'].notna()

df['next_valid_index'] = df['is_valid'].cumsum()
df_valid = df[df['is_valid']].copy()

df = df.groupby('next_valid_index').agg({
    'archivo_id': 'first',
    'Zulu Time': 'first',
    'Value': 'last',
    'bolus': 'sum',
    'carb': 'sum',
    'basal_rate': 'mean'
}).reset_index(drop=True)

df = df.dropna()
df = df.set_index('Zulu Time')

df = (
    df
    .groupby('archivo_id')
    .resample('5min')
    .agg({
        'Value': 'mean',
        'bolus': 'sum',
        'basal_rate': 'mean',
        'carb': 'sum'
    })
    .reset_index()
)

df = df.dropna()
df['diff'] = df.groupby('archivo_id')['Zulu Time'].diff().dt.total_seconds().div(60).astype('Int64')
df['archivo_id_diff'] = df['archivo_id'] != df['archivo_id'].shift()

df['time_jump'] = df['diff'] >= 30

df['new_group'] = df['archivo_id_diff'] | df['time_jump']

df['group_id'] = df['new_group'].cumsum() -1

df.drop(columns=['archivo_id_diff', 'time_jump', 'new_group', 'diff'], inplace=True)

gss = GroupShuffleSplit(test_size=0.2, random_state=10)
train_idx, test_idx = next(gss.split(df, groups=df['group_id']))

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

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


df_train = interpolar_por_grupo(df_train, method='quadratic')

df_train["carb"] = df_train["carb"].fillna(0)
df_train["bolus"] = df_train["bolus"].fillna(0)
df_train["basal_rate"] = df_train["basal_rate"].ffill().bfill()

df_test = df_test.reset_index(drop=True)

def eliminar_saltos(df):
    grupos_filtrados = []

    for group_id, group in df.groupby('group_id'):
        group = group.sort_values('Zulu Time').reset_index(drop=True)
        diffs = group['Zulu Time'].diff().dt.total_seconds().div(60).fillna(5)

        if (diffs >= 6).any():
            primer_salto = diffs[diffs > 5].index[0]
            grupo_filtrado = group.loc[:primer_salto - 1]
        else:
            grupo_filtrado = group

        if not grupo_filtrado.empty:
            grupos_filtrados.append(grupo_filtrado)

    return pd.concat(grupos_filtrados).reset_index(drop=True)

df_test = eliminar_saltos(df_test)


