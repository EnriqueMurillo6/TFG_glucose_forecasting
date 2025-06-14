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
