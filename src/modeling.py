# src/modeling.py
import lightgbm as lgb
import joblib

def train_lgbm(X_train, y_train, feature_cols=None):

    # Train LightGBM pake full dataset (X_train, y_train)
    model = lgb.LGBMRegressor(
        objective='poisson',
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_metric='mae',
        categorical_feature=['cluster','material','store'])

    # Save model
    joblib.dump(model, 'artifacts/lgbm_model.pkl')

    return model
