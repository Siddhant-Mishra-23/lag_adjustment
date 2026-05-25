import gc
import json, os, joblib, time
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

def prepare_dataframe():
    print("\n" + "=" * 75)
    print("Preparing the Dataframe")
    print("=" * 75 + "\n")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["lat", "lon", "date"]).reset_index(drop=True)
    #df_test = df[df["date"] > '2025-01-01']
    df = df[df["date"] < '2025-01-01']
    full_date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq="D")
    missing_reports = []
    print("\n" + "=" * 75)
    print("Checking the Missing Report")
    print("=" * 75 + "\n")
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
    df["loc_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str)
    df = df.sort_values(["loc_id", "date"]).reset_index(drop=True)
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
    df.dropna(subset=["sm_4_prior", "sm_3_prior", "sm_2_prior", "sm_1_prior", "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1", "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1"], inplace=True)
    
    return df

def get_model():
    models = {}
    models["RandomForest"] = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1)
    models["LIGHTGBM"] = lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, min_child_samples=10, random_state=SEED ,force_col_wise=True, verbose=-1, n_jobs=-1)
    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=5000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bynode=0.7, reg_alpha=0.1,
        reg_lambda=1.0, random_state=SEED, early_stopping_rounds=50, n_jobs=-1, verbosity=0, device="cuda")
    models["CatBoost"] = cb.CatBoostRegressor(
        iterations=5000, depth=8, learning_rate=0.01,
        l2_leaf_reg=3.0, random_seed=SEED, early_stopping_rounds=50, verbose=0,task_type="GPU")
    models["Ridge"] = Pipeline([
        ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    models["ElasticNet"] = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,random_state=SEED))])

    # base = []
    # base.append(("xgb", xgb.XGBRegressor(
    #     n_estimators=5000, max_depth=8, learning_rate=0.01,
    #     subsample=0.8, colsample_bynode=0.7, reg_alpha=0.1,
    #     reg_lambda=1.0, random_state=SEED, early_stopping_rounds=50, n_jobs=-1, verbosity=0, device="cuda")))
    # base.append(("cb", cb.CatBoostRegressor(
    #     iterations=5000, depth=8, learning_rate=0.01,
    #     l2_leaf_reg=3.0, random_seed=SEED, early_stopping_rounds=50, verbose=0,task_type="GPU")))
    # base.append(("rf", RandomForestRegressor(
    #     n_estimators=200, max_depth=15,
    #     random_state=SEED, n_jobs=-1)))
    # base.append(("lgb", lgb.LGBMRegressor(
    #     n_estimators=500, max_depth=8, learning_rate=0.05,
    #     subsample=0.8, min_samples_leaf=10, random_state=SEED)))
    # models["StackingEnsemble"] = StackingRegressor(
    #     estimators=base, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    return models

def compute_metrics(yt, yp):
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    bias = np.mean(yp - yt)
    ubrmse = np.sqrt(max(rmse**2 - bias**2, 0))
    return dict(rmse=rmse, mae=mae, r2=r2, ubrmse=ubrmse, bias=bias)

def get_cv_splits(df, cv_type):
    years = df["year"].unique()
    splits = []
    if cv_type == "groupkfold":
        gfk = GroupKFold(n_splits=min(5,len(years)))
        for tr, te in gfk.split(df, groups=df["year"].values):
            ty_ = df.iloc[te]["year"].unique()
            try_ = df.iloc[tr]["year"].unique()
            splits.append((tr, te, f"Train {try_} -> Test {ty_}"))
    elif cv_type == "timeseries_year":
        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]
            tr_mask = df["year"].isin(train_years)
            te_mask = df["year"] == test_year
            if len(df[tr_mask]) > 0 and len(df[te_mask]) > 0:
                splits.append((tr_mask, te_mask, f"Train {train_years} -> Test {test_year}"))  
    return splits        
        
def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))

def _best_fold_index(folds, metric):
    if not folds:
        raise ValueError("No fold metrics found.")
    if metric not in folds[0]:
        available = ", ".join(folds[0].keys())
        raise ValueError(f"Metric '{metric}' not found in folds. Available: {available}")
    values = np.asarray([fold[metric] for fold in folds], dtype=float)
    if metric == "r2":
        return int(np.nanargmax(values))
    if metric == "bias":
        return int(np.nanargmin(np.abs(values)))
    return int(np.nanargmin(values))

def save_model_outputs(mname, cv_type, folds, predictions, error_metric="rmse"):
    """Save best-fold train/valid parquet for one model right after training."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_idx = _best_fold_index(folds, error_metric)
    best_fold = best_idx + 1
    best_pred = predictions[best_idx]
    safe_model = _safe_name(mname)
    safe_cv = _safe_name(cv_type)

    train_df = pd.DataFrame(
        {
            "model": mname,
            "cv_type": cv_type,
            "fold": best_fold,
            "y_train": np.asarray(best_pred["y_train"]),
            "y_pred_train": np.asarray(best_pred["y_pred_train"]),
        }
    )
    valid_df = pd.DataFrame(
        {
            "model": mname,
            "cv_type": cv_type,
            "fold": best_fold,
            "y_test": np.asarray(best_pred["y_test"]),
            "y_pred_test": np.asarray(best_pred["y_pred_test"]),
        }
    )

    train_path = os.path.join(OUTPUT_DIR, f"{safe_model}_{safe_cv}_train.parquet")
    valid_path = os.path.join(OUTPUT_DIR, f"{safe_model}_{safe_cv}_valid.parquet")
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    print(
        f"Saved best-fold parquet for {mname} ({cv_type}) on fold {best_fold} "
        f"using {error_metric}."
    )


def run_experiment(X, y, df, cv_type, models, error_metric="rmse"):
    splits = get_cv_splits(df, cv_type)
    results = {}
    for mname, mtemplate in models.items():
        print("\n" + "=" * 75)
        print(f"Running {mname}")
        print("=" * 75 + "\n")

        folds = []
        predictions = []
        t0 = time.time()
        for f1, (tr_i, te_i, desc) in enumerate(splits, 1):
            print(f"Training {mname}, fold={f1}, label={desc}")
            model = clone(mtemplate)
            if hasattr(model, "early_stopping_rounds"):
                model.fit(X[tr_i], y[tr_i], eval_set=[(X[te_i], y[te_i])], verbose=False)
            else:
                model.fit(X[tr_i], y[tr_i])

            yp_test = model.predict(X[te_i])
            yp_train = model.predict(X[tr_i])
            metrics = compute_metrics(y[te_i], yp_test)
            metrics.update(fold=f1, label=desc, n_train=len(tr_i), n_test=len(te_i))
            folds.append(metrics)
            predictions.append(
                {
                    "y_test": np.asarray(y[te_i]),
                    "y_pred_test": np.asarray(yp_test),
                    "y_train": np.asarray(y[tr_i]),
                    "y_pred_train": np.asarray(yp_train),
                }
            )

        elapsed = time.time() - t0
        avg = {k: np.mean([f[k] for f in folds]) for k in ["rmse", "mae", "r2", "ubrmse", "bias"]}
        print(
            f"  Avg RMSE={avg['rmse']:.4f}  MAE={avg['mae']:.4f}  "
            f"R2={avg['r2']:.4f}  ubRMSE={avg['ubrmse']:.4f}  "
            f"Bias={avg['bias']:.4f}  [{elapsed:.1f}s]"
        )

        save_model_outputs(mname, cv_type, folds, predictions, error_metric=error_metric)
        results[mname] = {
            "folds": folds,
            "average": avg,
            "elapsed": elapsed,
        }

        del predictions
        gc.collect()
    return results
def print_summary(rg, rt):
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY")
    print("=" * 75)
    hdr = f"{'Model':<28}{'CV':<16}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'ubRMSE':>8}{'Bias':>8}{'Time':>7}"
    print(hdr); print("-"*len(hdr))
    for cn, res in [("GroupKFold", rg), ("TSplit", rt)]:
        for mn, r in res.items():
            a = r["average"]
            print(f"  {mn:<26}{cn:<16}{a['rmse']:>8.4f}{a['mae']:>8.4f}"
                  f"{a['r2']:>8.4f}{a['ubrmse']:>8.4f}{a['bias']:>8.4f}"
                  f"{r['elapsed']:>6.1f}s")
        print("-"*len(hdr))
    all_r2 = {}
    for cv, res in [("GKF",rg),("TS",rt)]:
        for n,r in res.items(): all_r2[f"{n} ({cv})"] = r["average"]["r2"]
    best = max(all_r2, key=all_r2.get)
    print(f"\n  * Best: {best}  (R2 = {all_r2[best]:.4f})")

def plot_results(results, cv_type, output_dir):
    """Generate and save comparison plots for experiment results."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    model_names = list(results.keys())
    avgs = {m: results[m]["average"] for m in model_names}

    def _point_density(x, y, bins=60):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0:
            return np.array([], dtype=float)
        if np.isclose(x.min(), x.max()) or np.isclose(y.min(), y.max()):
            return np.ones_like(x, dtype=float)
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
        y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
        return hist[x_bin, y_bin]

    # --- 1. Bar chart: metric comparison across models ---
    metrics_to_plot = ["rmse", "mae", "r2", "ubrmse"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    for ax, metric in zip(axes, metrics_to_plot):
        vals = [avgs[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Model Comparison - {cv_type}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"metric_comparison_{cv_type}.png"), dpi=200)
    plt.close(fig)

    # --- 2. Scatter: actual vs predicted (best fold per model) ---
    n = len(model_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for idx, mname in enumerate(model_names):
        ax = axes[idx // cols][idx % cols]
        valid_path = os.path.join(output_dir, f"{_safe_name(mname)}_{_safe_name(cv_type)}_valid.parquet")
        if not os.path.exists(valid_path):
            ax.set_visible(False)
            continue
        pred_df = pd.read_parquet(valid_path, columns=["y_test", "y_pred_test"])
        yt = pred_df["y_test"].to_numpy()
        yp = pred_df["y_pred_test"].to_numpy()
        density = _point_density(yt, yp)
        order = np.argsort(density)
        ax.scatter(
            yt[order],
            yp[order],
            c=density[order],
            cmap="coolwarm",
            alpha=0.75,
            s=10,
            edgecolors="none",
        )
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{mname}\n(R2={avgs[mname]['r2']:.4f})", fontsize=11)
    # hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle(f"Actual vs Predicted - {cv_type} (best fold)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"scatter_actual_vs_pred_{cv_type}.png"), dpi=200)
    plt.close(fig)

    # --- 3. Residual distribution (best fold per model) ---
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for idx, mname in enumerate(model_names):
        ax = axes[idx // cols][idx % cols]
        valid_path = os.path.join(output_dir, f"{_safe_name(mname)}_{_safe_name(cv_type)}_valid.parquet")
        if not os.path.exists(valid_path):
            ax.set_visible(False)
            continue
        pred_df = pd.read_parquet(valid_path, columns=["y_test", "y_pred_test"])
        residuals = pred_df["y_pred_test"].to_numpy() - pred_df["y_test"].to_numpy()
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Pred - Actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"{mname}\n(Bias={avgs[mname]['bias']:.4f})", fontsize=11)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle(f"Residual Distribution - {cv_type} (best fold)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"residual_distribution_{cv_type}.png"), dpi=200)
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}")


def save_results(
    results,
    cv_type,
    fname="all_results.json",
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fpath = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            all_records = json.load(f)
    else:
        all_records = []

    for mname, mdata in results.items():
        all_records.append(
            {
                "model": mname,
                "cv_type": cv_type,
                "average": mdata["average"],
                "elapsed": mdata["elapsed"],
                "folds": mdata["folds"],
            }
        )

    with open(fpath, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved {len(results)} result(s) to {fpath} (total: {len(all_records)})")
# def train_final(X, y, models, fname="all_results.json"):
#     # Load all results
#     fpath = os.path.join(OUTPUT_DIR, fname)
#     with open(fpath, "r") as f:
#         all_records = json.load(f)
#     # Group by model - score across all CV types
#     model_scores = defaultdict(list)
#     for rec in all_records:
#         avg_rmse = rec["average"]["rmse"]
#         avg_r2 = rec["average"]["r2"]
#         rmse_std = np.std([f["rmse"] for f in rec["folds"]])
#         score = avg_rmse + rmse_std - avg_r2
#         model_scores[rec["model"]].append({
#             "cv_type": rec["cv_type"],
#             "avg_rmse": avg_rmse,
#             "avg_r2": avg_r2,
#             "rmse_std": rmse_std,
#             "score": score,
#             "record": rec,
#         })
#     # Print summary
#     print("\n" + "=" * 75)
#     print("  Model Evaluation Summary")
#     print("=" * 75)
#     print(f"  {'Model':<20} {'CV Type':<20} {'RMSE':>8} {'R2':>8} {'RMSE_std':>10} {'Score':>8}")
#     print("  " + "-" * 75)

#     combined = {}
#     for mname, entries in model_scores.items():
#         for e in entries:
#             print(f"  {mname:<20} {e['cv_type']:<20} {e['avg_rmse']:>8.4f} {e['avg_r2']:>8.4f} {e['rmse_std']:>10.4f} {e['score']:>8.4f}")
#         avg_score = np.mean([e["score"] for e in entries])
#         combined[mname] = avg_score
#         print(f"  {mname:<20} {'** COMBINED **':<20} {'':>8} {'':>8} {'':>10} {avg_score:>8.4f}")
#         print()

#     # Best = lowest combined score
#     best_name = min(combined, key=combined.get)
#     print("  " + "-" * 75)
#     print(f"  Winner: {best_name} (combined score={combined[best_name]:.4f})")

#     m = clone(models[best_name])
#     m.fit(X, y)

#     # Save model
#     p = os.path.join(OUTPUT_DIR, "lag_adjustment_model.pkl")
#     joblib.dump(m, p)
#     print(f"  Saved '{best_name}' -> {p}")

#     # Save evaluation summary as CSV
#     summary_rows = []
#     for mname, entries in model_scores.items():
#         for e in entries:
#             summary_rows.append({
#                 "model": mname,
#                 "cv_type": e["cv_type"],
#                 "avg_rmse": e["avg_rmse"],
#                 "avg_mae": e["record"]["average"]["mae"],
#                 "avg_r2": e["avg_r2"],
#                 "avg_ubrmse": e["record"]["average"]["ubrmse"],
#                 "avg_bias": e["record"]["average"]["bias"],
#                 "rmse_std": e["rmse_std"],
#                 "score": e["score"],
#                 "combined_score": combined[mname],
#                 "is_best": mname == best_name,
#             })
#     summary_df = pd.DataFrame(summary_rows).sort_values("combined_score")
#     csv_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
#     summary_df.to_csv(csv_path, index=False)
#     print(f"  Summary saved -> {csv_path}")

#     return m, best_name


def train_all_models(X, y, models, output_dir):
    """Train every model on full data and save each as a .pkl for later testing."""
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for mname, mtemplate in models.items():
        print(f"  Training {mname} on full dataset ...")
        m = clone(mtemplate)
        if hasattr(m, "early_stopping_rounds"):
            m.set_params(early_stopping_rounds=None)
        m.fit(X, y)
        pkl_path = os.path.join(output_dir, f"{mname}.pkl")
        joblib.dump(m, pkl_path)
        saved[mname] = pkl_path
        print(f"    Saved -> {pkl_path}")
    print(f"\n  All {len(saved)} models saved to {output_dir}")
    return saved


if __name__ == "__main__":
    DATA_PATH = "/workspace/Siddhant/datasets/agriculture/odisha_merged_tabular.parquet"
    OUTPUT_DIR = "/workspace/Siddhant/agriculture/Output/"
    OUTPUT_DIR_Model = "/workspace/Siddhant/agriculture/Output/Models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_Model, exist_ok=True)
    SEED = 42
    FEAT = ["sm_4_prior", "sum_rainfall_4", "sum_rainfall_3", "sum_rainfall_2", "sum_rainfall_1", "mean_temp_4", "mean_temp_3", "mean_temp_2", "mean_temp_1", "doy_sin", "doy_cos"]
    TARGET = "soil_moisture"
    models = get_model()
    df = prepare_dataframe()
    X = df[FEAT].values
    y = df[TARGET].values
    r_groupkfold = run_experiment(X, y, df, "groupkfold", models)
    save_results(results=r_groupkfold , cv_type='groupkfold')
    plot_results(r_groupkfold, "groupkfold", OUTPUT_DIR)
    r_timeseries = run_experiment(X, y, df, "timeseries_year", models)
    save_results(results=r_timeseries , cv_type='timeseries')
    plot_results(r_timeseries, "timeseries", OUTPUT_DIR)
    print_summary(r_groupkfold,r_timeseries)
    # train_final(X, y, models)
    train_all_models(X, y, models, OUTPUT_DIR_Model)

