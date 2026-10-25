"""
Train LSTM, SARIMA, and XGBoost on India 2017 building data.
Train: Jan–Jun 2017 | Test: Jul–Dec 2017
Saves model artifacts for upload validation + model_results.json for dashboard.

Usage:
    cd backend
    pip install -r requirements.txt
    python train_models.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUT_PATH = os.path.join(ROOT, "model_results.json")

LSTM_WINDOW = 24
LSTM_EPOCHS = 20
SARIMA_MAX_TRAIN = 1000
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── Metrics ───────────────────────────────────────────────────────

def smape(y_true, y_pred):
    a, p = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = np.abs(a) + np.abs(p)
    mask = denom > 0
    return round(float(np.mean(2 * np.abs(a[mask] - p[mask]) / denom[mask]) * 100), 2)


def calc_metrics(y_true, y_pred):
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "sMAPE": smape(y_true, y_pred),
    }


# ── Data Loading ──────────────────────────────────────────────────

def load_india_2017():
    """Load all CSVs from data/ folder, combine W1+W2+W3, convert to IST,
    resample hourly, filter 2017, clean and add features."""
    print("[1/5] Loading India building data …")
    csv_files = sorted(
        [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    )

    df_list = []
    for path in csv_files:
        try:
            tmp = pd.read_csv(path, usecols=["TS", "W1", "W2", "W3"])
            tmp["W"] = tmp["W1"] + tmp["W2"] + tmp["W3"]
            df_list.append(tmp[["TS", "W"]])
        except Exception:
            pass

    full = pd.concat(df_list, ignore_index=True)
    full["datetime"] = (
        pd.to_datetime(full["TS"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Asia/Kolkata")
    )
    full = full.drop(columns="TS").set_index("datetime").sort_index()

    # Resample hourly & filter 2017
    hourly = full.resample("h").sum()
    df = hourly.loc["2017-01-01":"2017-12-31"].copy()

    # Outlier cap at mean + 3σ
    upper = df["W"].mean() + 3 * df["W"].std()
    df["W"] = df["W"].clip(upper=upper)

    # Gap fill — zero = sensor dropout
    df["W"] = df["W"].replace(0, np.nan).interpolate("linear")
    df["W"] = df["W"].fillna(df["W"].median())

    # Feature engineering
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print(f"    2017 shape: {df.shape} | W: {df['W'].min():.0f}–{df['W'].max():.0f} Wh")
    return df


# ── EDA ───────────────────────────────────────────────────────────

def compute_eda(df):
    """Compute EDA from full 2017 data."""
    print("[2/5] Computing EDA …")
    W = df["W"]

    hg = W.groupby(df.index.hour)
    hourly_profile = {
        "hours": list(range(24)),
        "labels": [f"{h:02d}:00" for h in range(24)],
        "load": [round(float(hg.mean()[h]), 1) for h in range(24)],
        "std": [round(float(hg.std()[h]), 1) for h in range(24)],
    }

    dg = W.groupby(df.index.dayofweek)
    daily_profile = {
        "days": DOW_NAMES,
        "load": [round(float(dg.mean()[i]), 1) for i in range(7)],
    }

    mg = W.groupby(df["month"])
    avail = sorted(mg.mean().index)
    monthly_profile = {
        "months": [MONTH_NAMES[m - 1] for m in avail],
        "load": [round(float(mg.mean()[m]), 1) for m in avail],
        "std": [round(float(mg.std()[m]), 1) for m in avail],
    }

    daily_trend = W.resample("D").mean()
    yearly_daily = {
        "labels": [d.strftime("%Y-%m-%d") for d in daily_trend.index],
        "actual": [round(float(v), 1) for v in daily_trend.values],
    }

    step = max(1, len(df) // 500)
    yearly_trend = {
        "labels": [d.strftime("%Y-%m-%d %H:00") for d in df.index[::step]],
        "actual": [round(float(v), 1) for v in W.values[::step]],
    }

    hm_data = []
    for m in avail:
        row = []
        md = df[df["month"] == m]
        for h in range(24):
            vals = md[md.index.hour == h]["W"]
            row.append(round(float(vals.mean()) if len(vals) else 0, 1))
        hm_data.append(row)
    heatmap = {
        "months": [MONTH_NAMES[m - 1] for m in avail],
        "hours": [f"{h:02d}:00" for h in range(24)],
        "data": hm_data,
    }

    wd = df[df["is_weekend"] == 0]
    we = df[df["is_weekend"] == 1]
    weekday_weekend = {
        "hours": list(range(24)),
        "labels": [f"{h:02d}:00" for h in range(24)],
        "weekday": [round(float(wd[wd.index.hour == h]["W"].mean()), 1) for h in range(24)],
        "weekend": [round(float(we[we.index.hour == h]["W"].mean()), 1) for h in range(24)],
    }

    train_df = df.loc["2017-01-01":"2017-06-30"]
    test_df = df.loc["2017-07-01":"2017-12-31"]
    peak_m = int(mg.mean().idxmax())
    low_m = int(mg.mean().idxmin())

    statistics = {
        "dataset": "IIT Bombay Residential Buildings (India, 2017)",
        "total_hours": int(len(df)),
        "train_hours": int(len(train_df)),
        "test_hours": int(len(test_df)),
        "train_period": "Jan 01, 2017 – Jun 30, 2017",
        "test_period": "Jul 01, 2017 – Dec 31, 2017",
        "W_mean": round(float(W.mean()), 1),
        "W_std": round(float(W.std()), 1),
        "W_min": round(float(W.min()), 1),
        "W_max": round(float(W.max()), 1),
        "W_median": round(float(W.median()), 1),
        "peak_hour": int(hg.mean().idxmax()),
        "lowest_hour": int(hg.mean().idxmin()),
        "peak_month": MONTH_NAMES[peak_m - 1],
        "lowest_month": MONTH_NAMES[low_m - 1],
        "weekday_avg": round(float(wd["W"].mean()), 1),
        "weekend_avg": round(float(we["W"].mean()), 1),
        "features_used": ["W (total power)", "hour_sin", "hour_cos",
                          "month_sin", "month_cos", "is_weekend", "lag features"],
        "lstm_window": LSTM_WINDOW,
    }

    return {
        "hourly": hourly_profile,
        "daily": daily_profile,
        "monthly": monthly_profile,
        "yearly": yearly_trend,
        "yearly_daily": yearly_daily,
        "heatmap": heatmap,
        "weekday_weekend": weekday_weekend,
        "statistics": statistics,
    }


# ── LSTM (disabled for now) ────────────────────────────────────────
# LSTM training is commented out to speed up iteration.
# To re-enable, uncomment train_lstm() and its call in main().

# def train_lstm(train_vals, test_vals):
#     print("[4/7] Training LSTM …")
#     import torch
#     import torch.nn as nn
#     scaler = MinMaxScaler()
#     train_sc = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
#     W = LSTM_WINDOW
#     X, y = [], []
#     for i in range(W, len(train_sc)):
#         X.append(train_sc[i - W:i])
#         y.append(train_sc[i])
#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.float32)
#     X_t = torch.from_numpy(X).unsqueeze(-1)
#     y_t = torch.from_numpy(y).unsqueeze(-1)
#     BATCH = 32
#     class LSTMModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.lstm1 = nn.LSTM(1, 64, batch_first=True)
#             self.drop1 = nn.Dropout(0.2)
#             self.lstm2 = nn.LSTM(64, 32, batch_first=True)
#             self.drop2 = nn.Dropout(0.2)
#             self.fc = nn.Linear(32, 1)
#         def forward(self, x):
#             x, _ = self.lstm1(x)
#             x = self.drop1(x)
#             x, _ = self.lstm2(x)
#             x = self.drop2(x[:, -1, :])
#             return self.fc(x)
#     device = torch.device("cpu")
#     model = LSTMModel().to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     best_loss = float("inf")
#     best_state = None
#     n_samples = len(X_t)
#     for epoch in range(LSTM_EPOCHS):
#         model.train()
#         perm = torch.randperm(n_samples)
#         epoch_loss = 0
#         n_batches = 0
#         for start in range(0, n_samples, BATCH):
#             idx = perm[start:start + BATCH]
#             xb = X_t[idx].to(device)
#             yb = y_t[idx].to(device)
#             optimizer.zero_grad()
#             pred = model(xb)
#             loss = criterion(pred, yb)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             n_batches += 1
#         avg_loss = epoch_loss / n_batches
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             best_state = {k: v.clone() for k, v in model.state_dict().items()}
#         print(f"    Epoch {epoch + 1}/{LSTM_EPOCHS}  loss={avg_loss:.6f}")
#     model.load_state_dict(best_state)
#     model.eval()
#     all_vals = np.concatenate([train_vals, test_vals])
#     all_sc = scaler.transform(all_vals.reshape(-1, 1)).flatten()
#     test_start = len(train_vals)
#     total_preds = len(test_vals)
#     X_test = np.array([all_sc[i - W:i] for i in range(test_start, len(all_vals))],
#                       dtype=np.float32)
#     X_test_t = torch.from_numpy(X_test).unsqueeze(-1).to(device)
#     pred_list = []
#     PRED_BATCH = 256
#     with torch.no_grad():
#         for start in range(0, len(X_test_t), PRED_BATCH):
#             batch = X_test_t[start:start + PRED_BATCH]
#             out = model(batch).squeeze(-1).cpu().numpy()
#             pred_list.append(out)
#             print(f"    Predicted {min(start + PRED_BATCH, total_preds)}/{total_preds}")
#     preds = np.concatenate(pred_list)
#     preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
#     preds = np.maximum(preds, 0)
#     os.makedirs(MODELS_DIR, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
#     with open(os.path.join(MODELS_DIR, "lstm_scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
#     print(f"    {len(preds)} predictions | model saved")
#     return preds


# ── SARIMA ────────────────────────────────────────────────────────

def train_sarima(train_vals, n_forecast):
    print("[4/5] Training SARIMA …")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    n = min(len(train_vals), SARIMA_MAX_TRAIN)
    subset = train_vals[-n:]
    print(f"    Fitting on last {n} hours …")

    model = SARIMAX(
        subset,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300)
    print(f"    AIC = {result.aic:.0f}")

    preds = result.forecast(steps=n_forecast)
    preds = np.maximum(np.array(preds, dtype=float), 0)
    print(f"    {len(preds)} predictions")
    return preds


# ── XGBoost ───────────────────────────────────────────────────────

XGB_FEATURE_COLS = None  # set during training, saved for upload validation


def build_xgb_features(df, target_col="W"):
    """Build XGBoost features from an hourly DataFrame with a power column."""
    d = df[[target_col]].copy()
    d.columns = ["W"]

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        d[f"lag_{lag}"] = d["W"].shift(lag)
    for w in [6, 12, 24]:
        d[f"roll_mean_{w}"] = d["W"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["W"].shift(1).rolling(w).std()

    d["hour"] = df.index.hour
    d["dayofweek"] = df.index.dayofweek
    d["month"] = df.index.month
    d["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    d["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    d["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    return d.dropna()


def train_xgboost(df_2017, split_time):
    global XGB_FEATURE_COLS
    print("[3/5] Training XGBoost …")
    import xgboost as xgb

    df_feat = build_xgb_features(df_2017, target_col="W")

    feat_cols = [c for c in df_feat.columns if c != "W"]
    XGB_FEATURE_COLS = feat_cols
    X = df_feat[feat_cols]
    y = df_feat["W"]

    train_mask = df_feat.index < split_time
    test_mask = df_feat.index >= split_time

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[test_mask], y[test_mask])],
        verbose=100,
    )

    preds = model.predict(X[test_mask])
    preds = np.maximum(preds, 0)

    # Save model & feature column list
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
    with open(os.path.join(MODELS_DIR, "xgb_feature_cols.json"), "w") as f:
        json.dump(feat_cols, f)

    print(f"    {len(preds)} predictions | model saved")
    return preds, y[test_mask].values, df_feat.index[test_mask]


# ── Assemble results ─────────────────────────────────────────────

def build_predictions(test_df, lstm_preds, sarima_preds, xgb_preds,
                      xgb_actual, xgb_idx):
    print("[5/5] Assembling results …")

    test_vals = test_df["W"].values
    test_idx = test_df.index

    # Align XGBoost to test index
    xgb_s = pd.Series(xgb_preds, index=xgb_idx)
    xgb_aligned = xgb_s.reindex(test_idx).interpolate().ffill().bfill().values

    to_list = lambda a: [round(float(v), 2) for v in a]

    hourly_labels = [d.strftime("%Y-%m-%d %H:00") for d in test_idx]

    # Daily aggregation
    daily_df = pd.DataFrame({
        "actual": test_vals, "lstm": lstm_preds,
        "sarima": sarima_preds, "xgboost": xgb_aligned,
    }, index=test_idx)
    daily_agg = daily_df.resample("D").mean().dropna()

    daily_pred = {
        "labels": [d.strftime("%b %d") for d in daily_agg.index],
        "actual": to_list(daily_agg["actual"]),
        "lstm": to_list(daily_agg["lstm"]),
        "sarima": to_list(daily_agg["sarima"]),
        "xgboost": to_list(daily_agg["xgboost"]),
    }

    # Monthly aggregation
    monthly_df = daily_df.copy()
    monthly_df["month"] = monthly_df.index.month
    mg = monthly_df.groupby("month").mean()
    test_months = [MONTH_NAMES[m - 1] for m in mg.index]

    monthly_pred = {
        "months": test_months,
        "actual": to_list(mg["actual"]),
        "lstm": to_list(mg["lstm"]),
        "sarima": to_list(mg["sarima"]),
        "xgboost": to_list(mg["xgboost"]),
    }

    # Monthly errors
    monthly_errors = {"months": test_months, "lstm": {}, "sarima": {}, "xgboost": {}}
    for m_num in mg.index:
        mask = test_idx.month == m_num
        m_name = MONTH_NAMES[m_num - 1]
        m_act = test_vals[mask]
        for model_key, m_pred in [("lstm", lstm_preds[mask]),
                                   ("sarima", sarima_preds[mask]),
                                   ("xgboost", xgb_aligned[mask])]:
            monthly_errors[model_key][m_name] = {
                "mae": round(float(mean_absolute_error(m_act, m_pred)), 0),
                "rmse": round(float(np.sqrt(mean_squared_error(m_act, m_pred))), 0),
            }

    # Weekly pattern
    weekly_pattern = {}
    for dow_i, dow_name in enumerate(DOW_NAMES):
        mask = test_idx.dayofweek == dow_i
        if not mask.any():
            continue
        dow_df = daily_df.loc[mask]
        hw = {}
        for col in ["actual", "lstm", "sarima", "xgboost"]:
            vals_by_hour = dow_df.groupby(dow_df.index.hour)[col].mean()
            hw[col] = [round(float(vals_by_hour.get(h, 0)), 1) for h in range(24)]
        weekly_pattern[dow_name] = hw

    return {
        "hourly_labels": hourly_labels,
        "actual": to_list(test_vals),
        "lstm": to_list(lstm_preds),
        "sarima": to_list(sarima_preds),
        "xgboost": to_list(xgb_aligned),
        "monthly": monthly_pred,
        "daily": daily_pred,
        "weekly_pattern": weekly_pattern,
        "monthly_errors": monthly_errors,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    df_2017 = load_india_2017()

    # Fixed split: Jan–Jun train, Jul–Dec test
    train_df = df_2017.loc["2017-01-01":"2017-06-30"]
    test_df = df_2017.loc["2017-07-01":"2017-12-31"]
    split_time = test_df.index[0]

    print(f"    Train: {train_df.index[0].date()} → {train_df.index[-1].date()}  "
          f"({len(train_df)} h)")
    print(f"    Test:  {test_df.index[0].date()} → {test_df.index[-1].date()}  "
          f"({len(test_df)} h)")

    train_vals = train_df["W"].values
    test_vals = test_df["W"].values

    # EDA (full 2017)
    eda = compute_eda(df_2017)

    # Save training data stats for upload validation scale-matching
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_stats = {
        "mean": float(train_vals.mean()),
        "std": float(train_vals.std()),
        "min": float(train_vals.min()),
        "max": float(train_vals.max()),
    }
    with open(os.path.join(MODELS_DIR, "train_stats.json"), "w") as f:
        json.dump(train_stats, f)

    # Train models — XGBoost first (before torch import for macOS libomp)
    xgb_preds, xgb_actual, xgb_idx = train_xgboost(df_2017, split_time)

    # LSTM — disabled for now
    # lstm_preds = train_lstm(train_vals, test_vals)
    lstm_preds = np.zeros(len(test_vals))  # placeholder

    # SARIMA
    sarima_preds = train_sarima(train_vals, len(test_vals))

    # Metrics
    xgb_s = pd.Series(xgb_preds, index=xgb_idx)
    xgb_aligned = xgb_s.reindex(test_df.index).interpolate().ffill().bfill().values

    metrics_data = {
        # LSTM disabled — placeholder metrics
        "lstm": {
            "R2": 0, "RMSE": 0, "MAE": 0, "sMAPE": 0,
            "label": "LSTM",
            "color": "#2563eb",
            "status": "disabled",
            "period": "(disabled)",
            "description": "LSTM training disabled — re-enable in train_models.py",
        },
        "sarima": {
            **calc_metrics(test_vals, sarima_preds),
            "label": "SARIMA",
            "status": "ok",
            "color": "#dc2626",
            "period": "Train: Jan–Jun 2017 | Test: Jul–Dec 2017",
            "description": (f"SARIMA(1,1,1)(1,1,1,24) — "
                            f"last {min(len(train_vals), SARIMA_MAX_TRAIN)}h training window"),
        },
        "xgboost": {
            **calc_metrics(test_vals, xgb_aligned),
            "label": "XGBoost",
            "status": "ok",
            "color": "#16a34a",
            "period": "Train: Jan–Jun 2017 | Test: Jul–Dec 2017",
            "description": "XGBoost with lag features (1h–168h), rolling stats, 500 trees, LR=0.05",
        },
    }

    # Predictions
    predictions = build_predictions(test_df, lstm_preds, sarima_preds,
                                    xgb_preds, xgb_actual, xgb_idx)

    # Save
    result = {"eda": eda, "metrics": metrics_data, "predictions": predictions}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n✓ Saved {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print(f"✓ Model artifacts saved in {MODELS_DIR}/")

    print("\nModel Metrics (Jul–Dec 2017):")
    for k in ["lstm", "sarima", "xgboost"]:
        m = metrics_data[k]
        print(f"  {m['label']:>8s}  R²={m['R2']:.4f}  RMSE={m['RMSE']:.1f}  "
              f"MAE={m['MAE']:.1f}  sMAPE={m['sMAPE']:.2f}%")


if __name__ == "__main__":
    main()
