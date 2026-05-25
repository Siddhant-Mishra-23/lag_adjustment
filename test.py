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
    return df

def prediction(df, model, feature_cols):
    start_time = time.time()
    X = df.loc[:, feature_cols].to_numpy(dtype=float)
    y_pred = model.predict(X)
    df["sm_pred"] = y_pred
    total_time = time.time() - start_time
    print(f"\nTotal prediction time: {total_time:.2f}s")
    return df

def evaluate_predictions(df, target_col):
    valid = df.dropna(subset=[target_col, "sm_pred"])
    mae = mean_absolute_error(valid[target_col], valid["sm_pred"]) 
    rmse = np.sqrt(mean_squared_error(valid[target_col], valid["sm_pred"])) 
    r2 = r2_score(valid[target_col], valid["sm_pred"])
    bias = (valid["sm_pred"] - valid[target_col]).mean()

    print(f"  Samples : {len(valid)}")
    if mae is not None:
        print(f"  MAE     : {mae:.4f}")
        print(f"  RMSE    : {rmse:.4f}")
        print(f"  R2      : {r2:.4f}")
        print(f"  Bias    : {bias:.4f}")

    return {
        "samples": int(len(valid)),
        "mae": float(mae) if mae is not None else None,
        "rmse": float(rmse) if rmse is not None else None,
        "r2": float(r2) if r2 is not None else None,
        "bias": float(bias) if bias is not None else None
    }


def save_outputs(df, metrics, output_path, model_name, target_col):
    os.makedirs(output_path, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in model_name)

    error_path = os.path.join(output_path, f"{safe_model}_error.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(output_path, f"{safe_model}_sm_pred.parquet")
    df_out = df[["date", "lat", "lon", target_col, "sm_pred"]].rename(
        columns={"lon": "lon", target_col: "sm"}
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
    MODEL_PATH = "/workspace/Siddhant/agriculture/Output/Models"   
    DATA_PATH = "/workspace/Siddhant/datasets/agriculture/odisha_merged_tabular.parquet"
    OUTPUT_PATH = "/workspace/Siddhant/agriculture/Output/Testing/"
    
    FEATURE_COLS = ["sm_4_prior", "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1", "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1", "doy_sin", "doy_cos"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)  # (a) mkdir exists
    df_base = prepare_dataframe()
    target_col = "soil_moisture"

    if os.path.isfile(MODEL_PATH):
        model_files = [MODEL_PATH]
    else:
        model_files = [
            os.path.join(MODEL_PATH, f)
            for f in sorted(os.listdir(MODEL_PATH))
            if f.lower().endswith((".pkl"))
        ]

    for model_file in model_files:  # (b) loop through all models
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        print(f"\nRunning model: {model_name}")
        df = df_base.copy()
        df["sm_pred"] = np.nan

        model = joblib.load(model_file)
        model_features = FEATURE_COLS

        df_predicted = prediction(df, model, model_features)
        metrics = evaluate_predictions(df_predicted, target_col)
        error_path, pred_path = save_outputs(df_predicted, metrics, OUTPUT_PATH, model_name, target_col)
        print(f"Wrote: {error_path}")   # (c.1) error json
        print(f"Wrote: {pred_path}")    # (c.2) sm & sm_pred parquet




