import pandas as pd

def load_full_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df['archivo_id'].isin(df['archivo_id'].unique()[:20])]
    return df
