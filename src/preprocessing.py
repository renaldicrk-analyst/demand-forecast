import pandas as pd

def filter_active_store(trx, days=30):
    trx = trx[trx['store'] != 'Cabang Testing'].copy()
    trx['trx_date'] = pd.to_datetime(trx['trx_date'])

    cutoff = trx['trx_date'].max() - pd.Timedelta(days=days)
    active_store = trx[trx['trx_date'] >= cutoff]['store'].unique()
    return trx[trx['store'].isin(active_store)]

def winsorize_entity(x):
    if x.nunique() < 5:
        return x
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return x.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

def apply_winsor(trx):
    trx['entity_id'] = trx['store'] + ' | ' + trx['material']
    trx['qty_w'] = trx.groupby('entity_id')['qty'].transform(winsorize_entity)
    return trx

def create_time_features(trx):
    
    # Buat semua fitur untuk forecasting: lag, rolling, calendar, dan categorical types
    # Sort dulu
    trx = trx.sort_values(['entity_id', 'trx_date'])

    # Lag features
    lags = [1, 7, 14]
    for l in lags:
        trx[f'lag_{l}'] = trx.groupby('entity_id')['qty'].shift(l)

    # Rolling features
    windows = [7, 14]
    for w in windows:
        trx[f'roll_mean_{w}'] = trx.groupby('entity_id')['qty'].shift(1).rolling(w).mean()
        trx[f'roll_std_{w}']  = trx.groupby('entity_id')['qty'].shift(1).rolling(w).std()

    # Calendar features
    trx['dow'] = trx['trx_date'].dt.weekday
    trx['is_weekend'] = trx['dow'].isin([5,6]).astype(int)
    trx['week'] = trx['trx_date'].dt.isocalendar().week.astype(int)

    # Categorical types
    trx['cluster'] = trx['cluster'].astype('category')
    trx['material'] = trx['material'].astype('category')
    trx['store'] = trx['store'].astype('category')

    return trx
