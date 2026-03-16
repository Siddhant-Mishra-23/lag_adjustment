"""
Vectorized Rolling Soil Moisture Prediction Pipeline
=====================================================
- Loads your pre-built feature dataframe (with all 14 feature columns)
- Loads your trained model
- Nulls out SM lags that fall inside the test period
- Iterates by DATE STEP (not by group) for speed
- Batch-predicts all (lat, lon) locations at each step
- Propagates predicted SM into future rows' lag columns
- Evaluates sm vs sm_pred
- Saves results
"""

import numpy as np
import pandas as pd
import joblib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG — Update these paths
# ============================================================
DATA_PATH = "your_feature_data.csv"       # your dataframe with all 14 features + sm
MODEL_PATH = "your_trained_model.pkl"      # trained XGBoost / LightGBM model
OUTPUT_PATH = "rolling_predictions.csv"    # where to save results

FEATURE_COLS = [
    "sm_4_prior", "sm_3_prior", "sm_2_prior", "sm_1_prior",
    "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1",
    "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1",
    "doy_sin", "doy_cos",
]


# ============================================================
# 1. LOAD DATA & MODEL
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
print(f"  Rows: {len(df)}")
print(f"  Locations: {df[['lat', 'long']].drop_duplicates().shape[0]}")
print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
print(f"  Dates: {df['date'].nunique()}")

print("\nLoading model...")
model = joblib.load(MODEL_PATH)
print(f"  Model type: {type(model).__name__}")


# ============================================================
# 2. ASSIGN TIME-STEP INDEX PER (lat, long) GROUP
# ============================================================
print("\nPreparing data...")
df = df.sort_values(["lat", "long", "date"]).reset_index(drop=True)
df["step"] = df.groupby(["lat", "long"]).cumcount()

n_steps = df["step"].max() + 1
n_locations = df[["lat", "long"]].drop_duplicates().shape[0]
print(f"  Time steps: {n_steps}")
print(f"  Locations:  {n_locations}")
print(f"  Total predictions to make: {len(df)}")


# ============================================================
# 3. NULL OUT SM LAGS THAT NEED PREDICTIONS
# ============================================================
#   step=0 (1st date): all 4 SM lags from history   -> keep all
#   step=1 (2nd date): sm_1_prior needs prediction   -> null it
#   step=2 (3rd date): sm_1, sm_2 need prediction    -> null them
#   step=3 (4th date): sm_1, sm_2, sm_3              -> null them
#   step>=4 (5th+):    all 4 SM lags                 -> null all
print("\nNulling SM lags that fall inside test period...")
for k in [1, 2, 3, 4]:
    df.loc[df["step"] >= k, f"sm_{k}_prior"] = np.nan

null_counts = {f"sm_{k}_prior": df[f"sm_{k}_prior"].isna().sum() for k in [1, 2, 3, 4]}
print(f"  Nulled counts: {null_counts}")


# ============================================================
# 4. ROLLING PREDICTION — ITERATE BY DATE STEP
# ============================================================
#   Loop over T time steps (not G groups).
#   At each step, model.predict() gets ALL locations as a batch.
#   Then propagate predictions into future rows' SM lag columns.
#
#   500 locations x 30 dates = 30 iterations (not 15,000)
# ============================================================
print("\nRunning rolling predictions...")
df["sm_pred"] = np.nan
start_time = time.time()

for t in range(n_steps):
    mask = df["step"] == t

    # --- Batch predict ALL locations for this time step ---
    X = df.loc[mask, FEATURE_COLS].values.astype(float)
    preds = model.predict(X)
    df.loc[mask, "sm_pred"] = preds

    # --- Propagate predictions into future SM lags ---
    #   Today's prediction at each (lat, lon) becomes:
    #     sm_1_prior at step t+1  (tomorrow)
    #     sm_2_prior at step t+2  (day after)
    #     sm_3_prior at step t+3
    #     sm_4_prior at step t+4
    for offset, col in enumerate(
        ["sm_1_prior", "sm_2_prior", "sm_3_prior", "sm_4_prior"], start=1
    ):
        future_step = t + offset
        if future_step >= n_steps:
            continue

        future_mask = df["step"] == future_step

        # Build lookup: (lat, lon) -> predicted value from current step
        pred_series = pd.Series(
            preds,
            index=pd.MultiIndex.from_frame(df.loc[mask, ["lat", "long"]]),
        )

        # Match future rows by their (lat, lon)
        future_idx = pd.MultiIndex.from_frame(df.loc[future_mask, ["lat", "long"]])
        fill_vals = pred_series.reindex(future_idx).values

        # Fill only NaN slots (don't overwrite actual history values)
        current_vals = df.loc[future_mask, col]
        df.loc[future_mask, col] = current_vals.where(current_vals.notna(), fill_vals)

    # Progress
    if (t + 1) % 10 == 0 or t == n_steps - 1:
        elapsed = time.time() - start_time
        print(f"  Step {t + 1}/{n_steps} done ({elapsed:.2f}s)")

total_time = time.time() - start_time
print(f"\nTotal prediction time: {total_time:.2f}s")


# ============================================================
# 5. VERIFY PROPAGATION
# ============================================================
print("\n" + "=" * 60)
print("PROPAGATION VERIFICATION (first location)")
print("=" * 60)
first_lat = df["lat"].iloc[0]
first_lon = df["long"].iloc[0]
sample = df[(df["lat"] == first_lat) & (df["long"] == first_lon)].head(8)

print(sample[[
    "date", "sm", "sm_pred",
    "sm_4_prior", "sm_3_prior", "sm_2_prior", "sm_1_prior"
]].to_string(index=False))

# Check: sm_pred of row N should equal sm_1_prior of row N+1
print("\nPropagation check (sm_pred[i] == sm_1_prior[i+1]):")
for i in range(min(5, len(sample) - 1)):
    pred_val = sample.iloc[i]["sm_pred"]
    lag_val = sample.iloc[i + 1]["sm_1_prior"]
    match = "pass" if np.isclose(pred_val, lag_val, atol=1e-6) else "FAIL"
    print(f"  Row {i} pred={pred_val:.6f}  ->  Row {i+1} sm_1_prior={lag_val:.6f}  [{match}]")


# ============================================================
# 6. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

# Overall
valid = df.dropna(subset=["sm", "sm_pred"])
mae = mean_absolute_error(valid["sm"], valid["sm_pred"])
rmse = np.sqrt(mean_squared_error(valid["sm"], valid["sm_pred"]))
r2 = r2_score(valid["sm"], valid["sm_pred"])

print(f"  Samples : {len(valid)}")
print(f"  MAE     : {mae:.4f}")
print(f"  RMSE    : {rmse:.4f}")
print(f"  R2      : {r2:.4f}")

# Per-step metrics (shows how error grows as more lags become predicted)
print("\nPer-step metrics (error accumulation over time):")
print(f"  {'Step':>5} {'Date':>12} {'Count':>7} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print(f"  {'-'*5} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

for t in range(min(n_steps, 20)):
    step_data = df[df["step"] == t].dropna(subset=["sm", "sm_pred"])
    if len(step_data) == 0:
        continue
    s_mae = mean_absolute_error(step_data["sm"], step_data["sm_pred"])
    s_rmse = np.sqrt(mean_squared_error(step_data["sm"], step_data["sm_pred"]))
    s_r2 = r2_score(step_data["sm"], step_data["sm_pred"]) if len(step_data) > 1 else float("nan")
    sample_date = step_data["date"].iloc[0].strftime("%Y-%m-%d")
    print(f"  {t:>5} {sample_date:>12} {len(step_data):>7} {s_mae:>8.4f} {s_rmse:>8.4f} {s_r2:>8.4f}")


# ============================================================
# 7. SAVE RESULTS
# ============================================================
df_out = df.drop(columns=["step"])
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"Columns in output: {list(df_out.columns)}")
print("Done!")