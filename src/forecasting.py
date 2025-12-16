import pandas as pd
import numpy as np

def build_features(cur, buffer):
    """
    Build lag, rolling, calendar, and categorical features untuk forecasting.
    cur: dataframe target (future date)
    buffer: dataframe historis (entity_id, trx_date, qty)
    """
    cur['entity_id'] = cur['entity_id'].astype(str)
    buffer['entity_id'] = buffer['entity_id'].astype(str)

    # LAG FEATURES
    for lag in [1, 7, 14]:
        cur[f'lag_{lag}'] = cur['entity_id'].map(
            buffer.groupby('entity_id')['qty']
                  .apply(lambda x: x.shift(lag).iloc[-1] if len(x) >= lag else 0)
        )

    # ROLLING FEATURES
    roll_data = []
    for eid, df in buffer.groupby('entity_id'):
        vals = df.sort_values('trx_date')['qty']
        roll_data.append([
            eid,
            vals.shift(1).rolling(7).mean().iloc[-1]  if len(vals) >= 7  else 0,
            vals.shift(1).rolling(7).std().iloc[-1]   if len(vals) >= 7  else 1,
            vals.shift(1).rolling(14).mean().iloc[-1] if len(vals) >= 14 else 0,
            vals.shift(1).rolling(14).std().iloc[-1]  if len(vals) >= 14 else 1,
        ])

    roll_df = pd.DataFrame(
        roll_data,
        columns=['entity_id','roll_mean_7','roll_std_7','roll_mean_14','roll_std_14']
    )

    cur = cur.merge(roll_df, on='entity_id', how='left')

    # CALENDAR FEATURES
    cur['dow'] = cur['trx_date'].dt.weekday
    cur['is_weekend'] = cur['dow'].isin([5, 6]).astype(int)
    cur['week'] = cur['trx_date'].dt.isocalendar().week.astype(int)

    # CATEGORICAL
    cur['cluster'] = cur['cluster'].astype('category')
    cur['material'] = cur['material'].astype('category')
    cur['store'] = cur['store'].astype('category')

    return cur


def forecast_7d(trx, model, feature_cols):
    """
    Forecast demand 7 hari ke depan per entity (natural forecast, tanpa buffer)
    """
    trx['entity_id'] = trx['entity_id'].astype(str)
    history = trx.sort_values(['entity_id', 'trx_date'])
    last_date = history['trx_date'].max()

    # Base entity table
    entities = history[['entity_id','store','material','cluster']].drop_duplicates()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7,
        freq='D'
    )

    future = (
        entities.assign(key=1)
        .merge(pd.DataFrame({'trx_date': future_dates, 'key': 1}), on='key')
        .drop('key', axis=1)
    )

    predictions = []

    # buffer historis + prediksi recursive
    buffer = history[['entity_id','trx_date','qty']].copy()

    for step in range(7):
        cur_date = last_date + pd.Timedelta(days=step + 1)
        cur = future[future['trx_date'] == cur_date].copy()

        # Build features
        cur = build_features(cur, buffer)

        # Predict
        X_cur = cur[feature_cols]
        # cur['forecast_qty'] = model.predict(X_cur).clip(lower=0).round().astype(int)
        y_pred = model.predict(X_cur)
        y_pred = np.clip(y_pred, 0, None)

        cur['forecast_qty'] = np.round(y_pred).astype(int)


        # Update buffer
        buffer = pd.concat([
            buffer,
            cur[['entity_id','trx_date','forecast_qty']]
                .rename(columns={'forecast_qty': 'qty'})
        ], ignore_index=True)

        predictions.append(cur)

    forecast_df = pd.concat(predictions, ignore_index=True)

    final_planning = (
        forecast_df[['trx_date','store','material','cluster','forecast_qty']]
        .sort_values(['store','trx_date','material'])
        .reset_index(drop=True)
    )

    return final_planning
