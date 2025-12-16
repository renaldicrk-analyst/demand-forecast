from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import pandas as pd
import joblib

def build_cluster_features(trx):
    feat = trx.groupby('entity_id').agg(
        median_qty=('qty_w','median'),
        p90_qty=('qty_w', lambda x: x.quantile(0.9)),
        std_qty=('qty_w','std'),
        mean_qty=('qty_w','mean'),
        active_days=('trx_date','nunique'))
    feat['cv'] = feat['std_qty'] / (feat['mean_qty'] + 1e-6)
    return feat.fillna(0)

def train_kmeans(feat, k=5):
    X = feat[['median_qty','p90_qty','cv','active_days']]
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    feat['cluster'] = kmeans.fit_predict(Xs)

    joblib.dump(scaler, 'artifacts/scaler.pkl')
    joblib.dump(kmeans, 'artifacts/kmeans.pkl')
    feat.to_parquet('artifacts/cluster_df.parquet')

    return feat
