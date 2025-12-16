# main.py
from dotenv import load_dotenv 
load_dotenv()
from src.data_loader import load_trx
from src.preprocessing import filter_active_store, apply_winsor, create_time_features
from src.clustering import build_cluster_features, train_kmeans
from src.modeling import train_lgbm
from src.forecasting import forecast_7d

# LOAD DATA
trx = load_trx()

# PREPROCESSING
trx = filter_active_store(trx)
trx = apply_winsor(trx)  # menghasilkan qty

# CLUSTERING
feat = build_cluster_features(trx)
cluster_df = train_kmeans(feat)

trx = trx.merge(
    cluster_df[['cluster']],
    left_on='entity_id',
    right_index=True
)

# TIME SERIES FEATURES
trx = create_time_features(trx)

# FEATURE COLUMNS
feature_cols = [
    'lag_1','lag_7','lag_14',
    'roll_mean_7','roll_std_7',
    'roll_mean_14','roll_std_14',
    'dow','is_weekend','week',
    'cluster','material','store'
]

# TRAIN MODEL
train_df = trx.dropna(subset=feature_cols + ['qty'])
X_train = train_df[feature_cols]
y_train = train_df['qty']

model = train_lgbm(X_train, y_train, feature_cols)

# FORECAST (NATURAL)
final_planning = forecast_7d(trx, model, feature_cols)

# SAVE OUTPUT
final_planning.to_parquet(
    "outputs/final_planning.parquet",
    index=False
)

print("== Forecast 7 hari  tersimpan di outputs/final_planning.parquet ==")
