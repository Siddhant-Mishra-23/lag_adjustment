import numpy as np
import pandas as pd
import joblib
import time
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_dataframe():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    #df["loc_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str)
    #df = df.sort_values(["loc_id", "date"]).reset_index(drop=True)
    df["sm_4_prior"] = df["soil_moisture"].shift(4)
    df["sm_3_prior"] = df["soil_moisture"].shift(3)
    df["sm_2_prior"] = df["soil_moisture"].shift(2)
    df["sm_1_prior"] = df["soil_moisture"].shift(1)
    df["sum_rainfall_4"] = sum(df["rainfall"].shift(i) for i in range(1, 5))
    df["sum_rainfall_3"] = sum(df["rainfall"].shift(i) for i in range(1, 4))
    df["sum_rainfall_2"] = sum(df["rainfall"].shift(i) for i in range(1, 3))
    df["sum_rainfall_1"] = sum(df["rainfall"].shift(i) for i in range(1, 2))
    df["mean_temp_4"] = sum(df["temperature"].shift(i) for i in range(1, 5)) / 4
    df["mean_temp_3"] = sum(df["temperature"].shift(i) for i in range(1, 4)) / 3
    df["mean_temp_2"] = sum(df["temperature"].shift(i) for i in range(1, 3)) / 2
    df["mean_temp_1"] = sum(df["temperature"].shift(i) for i in range(1, 2)) / 1
    df["doy"]     = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    #df.dropna(subset=["sm_4_prior", "sm_3_prior", "sm_2_prior", "sm_1_prior", "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1", "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1"], inplace=True)

    df = df[df["date"]>= "2025-01-01"].reset_index(drop=True) 
    df["step"] = df.groupby(["lat", "lon"]).cumcount()
    n_steps = df["step"].max() + 1
    n_locations = df[["lat", "lon"]].drop_duplicates().shape[0]
    full_date_range = pd.date_range(start='2025-01-01', end='2025-12-31', freq="D")
    missing_reports = []
    for (lat, lon), group in df.groupby(['lat', 'lon']):
        existing_dates = set(group['date'])
        missing_dates = set(full_date_range) - existing_dates
        if missing_dates != set():
            missing_reports.append({
                'lat': lat,
                'lon': lon,
                'missing_dates': sorted(missing_dates)
            })
    if missing_reports == []:
        print("No missing reports found.")
        in_sync = True
    else:
        print(f"Found {len(missing_reports)} locations with missing reports.")
        in_sync = False
    return df,n_steps,n_locations

def rolling_prediction(df, model, n_steps, feature_cols, lon_col):
    start_time = time.time()
    for t in range(n_steps):
        mask = df["step"] == t
        X = df.loc[mask, feature_cols].values.astype(float)
        y_pred = model.predict(X)
        df.loc[mask, "sm_pred"] = y_pred
        for offset, col in enumerate(["sm_4_prior"], start=1):
            future_step = t + offset
            if future_step >= n_steps:
                continue
            future_mask = df["step"] == future_step
            pred_series = pd.Series(
                y_pred, 
                index=pd.MultiIndex.from_frame(df.loc[mask, ["lat", lon_col]]),
            )
            future_idx = pd.MultiIndex.from_frame(df.loc[future_mask, ["lat", lon_col]])
            fill_vals = pred_series.reindex(future_idx).values

            # Fill only NaN slots (don't overwrite actual history values)
            current_vals = df.loc[future_mask, col]
            df.loc[future_mask, col] = current_vals.where(current_vals.notna(), fill_vals)
        if (t + 1) % 10 == 0 or t == n_steps - 1:
            elapsed = time.time() - start_time
            print(f"  Step {t + 1}/{n_steps} done ({elapsed:.2f}s)")

    total_time = time.time() - start_time
    print(f"\nTotal prediction time: {total_time:.2f}s")
    return df

def evaluate_predictions(df, target_col, n_steps):
    valid = df.dropna(subset=[target_col, "sm_pred"])
    mae = mean_absolute_error(valid[target_col], valid["sm_pred"]) if len(valid) else None
    rmse = np.sqrt(mean_squared_error(valid[target_col], valid["sm_pred"])) if len(valid) else None
    r2 = r2_score(valid[target_col], valid["sm_pred"]) if len(valid) else None

    print(f"  Samples : {len(valid)}")
    if mae is not None:
        print(f"  MAE     : {mae:.4f}")
        print(f"  RMSE    : {rmse:.4f}")
        print(f"  R2      : {r2:.4f}")

    step_metrics = []
    for t in range(min(n_steps, 20)):
        step_data = df[df["step"] == t].dropna(subset=[target_col, "sm_pred"])
        if len(step_data) == 0:
            continue
        s_mae = mean_absolute_error(step_data[target_col], step_data["sm_pred"])
        s_rmse = np.sqrt(mean_squared_error(step_data[target_col], step_data["sm_pred"]))
        s_r2 = r2_score(step_data[target_col], step_data["sm_pred"]) if len(step_data) > 1 else float("nan")
        sample_date = step_data["date"].iloc[0].strftime("%Y-%m-%d")
        print(f"  {t:>5} {sample_date:>12} {len(step_data):>7} {s_mae:>8.4f} {s_rmse:>8.4f} {s_r2:>8.4f}")
        step_metrics.append({
            "step": int(t),
            "date": sample_date,
            "count": int(len(step_data)),
            "mae": float(s_mae),
            "rmse": float(s_rmse),
            "r2": float(s_r2) if np.isfinite(s_r2) else None,
        })

    return {
        "samples": int(len(valid)),
        "mae": float(mae) if mae is not None else None,
        "rmse": float(rmse) if rmse is not None else None,
        "r2": float(r2) if r2 is not None else None,
        "per_step": step_metrics,
    }


def save_outputs(df, metrics, output_path, model_name, target_col, lon_col):
    os.makedirs(output_path, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in model_name)

    error_path = os.path.join(output_path, f"{safe_model}_error.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(output_path, f"{safe_model}_sm_pred.parquet")
    df_out = df[["date", "lat", lon_col, target_col, "sm_pred"]].rename(
        columns={lon_col: "lon", target_col: "sm"}
    )
    df_out.to_parquet(pred_path, index=False)
    return error_path, pred_path




if __name__ == "__main__":
    """
    This script performs a rolling prediction of soil moisture (sm) using a trained model.
    It evaluates the predictions against actual sm values and saves the results.
    
    Steps:
    1. Load data and model
    2. Assign time-step index per (lat, long) group
    3. Null out SM lags that need predictions
    4. Make rolling predictions
    5. Evaluate predictions vs actual sm
    6. Save results
    """
    MODEL_PATH = r"D:\AHRC\Irrigation_Git\lag_adjustment\Models"
    
    DATA_PATH = r"D:\AHRC\Irrigation_Git\lag_adjustment\final_output\odisha_merged_tabular.parquet"
    OUTPUT_PATH = r"D:\AHRC\Irrigation_Git\lag_adjustment\final_output\rolling_eval"
    
    FEATURE_COLS = [
    "sm_4_prior","sum_rainfall_4", "mean_temp_4", "mean_temp_3","doy_sin", "doy_cos",]

    os.makedirs(OUTPUT_PATH, exist_ok=True)  # (a) mkdir exists
    df_base, n_steps, n_locations = prepare_dataframe()
    lon_col = "lon" if "lon" in df_base.columns else "long"
    target_col = "sm" if "sm" in df_base.columns else "soil_moisture"

    if os.path.isfile(MODEL_PATH):
        model_files = [MODEL_PATH]
    else:
        model_files = [
            os.path.join(MODEL_PATH, f)
            for f in sorted(os.listdir(MODEL_PATH))
            if f.lower().endswith((".pkl", ".joblib"))
        ]

    for model_file in model_files:  # (b) loop through all models
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        print(f"\nRunning model: {model_name}")
        df = df_base.copy()
        print("Nulling SM lags that fall inside test period...")
        for k in [1, 2, 3, 4]:
            df.loc[df["step"] >= k, f"sm_{k}_prior"] = np.nan
        df["sm_pred"] = np.nan

        model = joblib.load(model_file)
        model_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else FEATURE_COLS
        missing = [c for c in model_features if c not in df.columns]
        if missing:
            metrics = {"status": "failed", "error": f"Missing features: {missing}"}
            error_path, _ = save_outputs(df, metrics, OUTPUT_PATH, model_name, target_col, lon_col)
            print(f"Skipped {model_name}. Wrote: {error_path}")
            continue

        df_rolled = rolling_prediction(df, model, n_steps, model_features, lon_col)
        metrics = evaluate_predictions(df_rolled, target_col, n_steps)
        error_path, pred_path = save_outputs(df_rolled, metrics, OUTPUT_PATH, model_name, target_col, lon_col)
        print(f"Wrote: {error_path}")   # (c.1) error json
        print(f"Wrote: {pred_path}")    # (c.2) sm & sm_pred parquet




