"""
=============================================================================
  Random Forest – Soil Moisture Prediction
=============================================================================
  Features (per lat-lon-date row):
    1. sm_lag4        : soil moisture at (Date − 4 days)
    2. rain_sum_4d    : Σ rainfall   for Date-1, -2, -3, -4
    3. temp_mean_4d   : μ temperature for Date-1, -2, -3, -4
    4. lat, lon       : location coordinates

  Target : soil_moisture on the given Date

  Cross-Validation : Year-based GroupKFold
      – each fold holds out ALL data for one year
      – n_splits = number of unique years (ideally 5)
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import warnings, textwrap
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  LOAD & INSPECT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATH = "/mnt/user-data/uploads/odisha_merged_tabular.csv"

print("=" * 70)
print("  STEP 1 :  Loading data")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"])
df.sort_values(["lat", "lon", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Rows            : {len(df):,}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Date range      : {df['date'].min().date()} -> {df['date'].max().date()}")
years_all = sorted(df["date"].dt.year.unique())
print(f"  Unique years    : {years_all}")
print(f"  Unique dates    : {df['date'].nunique()}")
print(f"  Unique (lat,lon): {df.groupby(['lat','lon']).ngroups:,}")
print(f"\n  Rows per year:")
print(df.groupby(df["date"].dt.year).size().to_string())
print(f"\n  Nulls:\n{df.isnull().sum().to_string()}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  FEATURE ENGINEERING  (per location group)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 2 :  Engineering lag features")
print("=" * 70)

def build_features(g):
    g = g.sort_values("date").copy()
    g["sm_lag4"]      = g["soil_moisture"].shift(4)
    g["rain_sum_4d"]  = (g["rainfall"].shift(1).fillna(0)
                         + g["rainfall"].shift(2).fillna(0)
                         + g["rainfall"].shift(3).fillna(0)
                         + g["rainfall"].shift(4).fillna(0))
    g["temp_mean_4d"] = (g["temperature"].shift(1)
                         + g["temperature"].shift(2)
                         + g["temperature"].shift(3)
                         + g["temperature"].shift(4)) / 4.0
    return g

# Check for CONSECUTIVE daily runs (not just date count)
def max_consecutive_days(group):
    dates = group["date"].sort_values().reset_index(drop=True)
    gaps  = dates.diff().dt.days.fillna(1)
    runs, current = [], 1
    for g in gaps.iloc[1:]:
        if g == 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return max(runs)

sample_loc = df.groupby(["lat", "lon"]).first().index[0]
sample_consec = max_consecutive_days(
    df[(df["lat"] == sample_loc[0]) & (df["lon"] == sample_loc[1])]
)
total_dates = df["date"].nunique()
print(f"  Total unique dates           : {total_dates}")
print(f"  Max consecutive days (sample): {sample_consec}")

USE_SYNTHETIC = False

if sample_consec < 5:
    print(f"\n  ⚠  DATA GAP WARNING")
    print(f"     Max consecutive daily dates per location: {sample_consec}")
    print(f"     Lag-4 features require ≥ 5 consecutive daily records.")
    print(f"     Current dates: {sorted(df['date'].dt.date.unique())}")
    print(f"\n     → Running demo on SYNTHETIC daily data to show full pipeline.")
    print(f"     → Once you have ≥5 days of daily data, this script works as-is.\n")
    USE_SYNTHETIC = True

    # ── Generate realistic synthetic Odisha-like data ──
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    lats  = np.round(np.linspace(20.5, 22.5, 4), 2)
    lons  = np.round(np.linspace(83.5, 86.0, 5), 2)

    rows = []
    for lat in lats:
        for lon in lons:
            n = len(dates)
            day_idx = np.arange(n)
            base_sm = 0.22 + 0.01 * (lat - 20) + 0.005 * (lon - 83)
            sm = (base_sm
                  + 0.08 * np.sin(day_idx * 2 * np.pi / 365)
                  + np.random.normal(0, 0.012, n)).clip(0.05, 0.55)
            rain = np.maximum(0, 8 * np.sin(day_idx * 2 * np.pi / 365 - 0.5)
                              + np.random.exponential(2, n))
            temp = (32 + 7 * np.sin(day_idx * 2 * np.pi / 365 + 1.2)
                    + np.random.normal(0, 1.8, n))
            for i, d in enumerate(dates):
                rows.append([d, lat, lon, sm[i], rain[i], temp[i]])

    df = pd.DataFrame(rows, columns=["date", "lat", "lon", "soil_moisture",
                                      "rainfall", "temperature"])
    df.sort_values(["lat", "lon", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Synthetic dataset: {len(df):,} rows, "
          f"{df['date'].dt.year.nunique()} years, "
          f"{df.groupby(['lat','lon']).ngroups} locations")

df["loc_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str)
df = df.groupby("loc_id", group_keys=False).apply(build_features)
if "loc_id" in df.columns:
    df.drop(columns=["loc_id"], inplace=True)

rows_before = len(df)
df.dropna(subset=["sm_lag4", "temp_mean_4d"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\n  Rows before NaN drop : {rows_before:,}")
print(f"  Rows after  NaN drop : {len(df):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  PREPARE X, y, YEAR GROUPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 3 :  Preparing features / target / year groups")
print("=" * 70)

FEATURE_COLS = ["sm_lag4", "rain_sum_4d", "temp_mean_4d", "lat", "lon"]
TARGET_COL   = "soil_moisture"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

df["year"] = df["date"].dt.year
years_unique = sorted(df["year"].unique())
n_years  = len(years_unique)
n_splits = min(5, n_years)
groups   = df["year"].values

print(f"  Feature matrix : {X.shape}")
print(f"  Years          : {years_unique}")
print(f"  Folds (splits) : {n_splits}  (1 year held out per fold)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  YEAR-BASED CROSS-VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 4 :  Year-based GroupKFold Cross-Validation")
print("=" * 70)

header = (f"{'Fold':>4}  {'Test Year(s)':<22}  {'Train':>8}  "
          f"{'Test':>8}  {'RMSE':>10}  {'MAE':>10}  {'R2':>10}")
print(header)
print("-" * len(header))

gkf = GroupKFold(n_splits=n_splits)
fold_results = []

for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    test_years = sorted(df.iloc[test_idx]["year"].unique())

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)

    fold_results.append({
        "fold": fold_i, "years": test_years,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "rmse": rmse, "mae": mae, "r2": r2,
        "y_test": y_te, "y_pred": y_pred,
    })

    print(f"{fold_i:>4}  {str(test_years):<22}  {len(train_idx):>8,}  "
          f"{len(test_idx):>8,}  {rmse:>10.4f}  {mae:>10.4f}  {r2:>10.4f}")

# ── Averages ──
avg_rmse = np.mean([r["rmse"] for r in fold_results])
avg_mae  = np.mean([r["mae"]  for r in fold_results])
avg_r2   = np.mean([r["r2"]   for r in fold_results])

print("-" * len(header))
print(f"{'AVG':>4}  {'':22}  {'':8}  {'':8}  "
      f"{avg_rmse:>10.4f}  {avg_mae:>10.4f}  {avg_r2:>10.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  FEATURE IMPORTANCE (final full model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 5 :  Final model on ALL data + Feature Importance")
print("=" * 70)

rf_final = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X, y)

importances = rf_final.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<15} {imp:.4f}  {bar}")

MODEL_PATH = "/home/claude/rf_soil_moisture_final.pkl"
joblib.dump(rf_final, MODEL_PATH)
print(f"\n  Final model saved -> {MODEL_PATH}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  PLOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  STEP 6 :  Generating diagnostic plots")
print("=" * 70)

data_label = "(Synthetic Demo)" if USE_SYNTHETIC else "(Real Data)"

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"Random Forest — Soil Moisture Prediction {data_label}\n"
             f"Year-Based {n_splits}-Fold Cross-Validation",
             fontsize=14, fontweight="bold", y=0.98)

# 6a. Scatter: Observed vs Predicted
ax = axes[0, 0]
all_yt = np.concatenate([r["y_test"] for r in fold_results])
all_yp = np.concatenate([r["y_pred"] for r in fold_results])
ax.scatter(all_yt, all_yp, alpha=0.12, s=6, color="steelblue", edgecolors="none")
mn, mx = min(all_yt.min(), all_yp.min()), max(all_yt.max(), all_yp.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="1:1 line")
ax.set_xlabel("Observed Soil Moisture")
ax.set_ylabel("Predicted Soil Moisture")
ax.set_title("Observed vs Predicted (all folds)")
ax.legend()
ax.text(0.05, 0.92, f"R² = {avg_r2:.3f}\nRMSE = {avg_rmse:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

# 6b. Per-fold metrics bar chart
ax = axes[0, 1]
folds_x = [f"Fold {r['fold']}\n{r['years']}" for r in fold_results]
rmses   = [r["rmse"] for r in fold_results]
r2s     = [r["r2"]   for r in fold_results]
x_pos = np.arange(len(folds_x))
w = 0.35
ax.bar(x_pos - w/2, rmses, w, label="RMSE", color="salmon")
ax_r = ax.twinx()
ax_r.bar(x_pos + w/2, r2s, w, label="R²", color="mediumseagreen")
ax.set_xticks(x_pos)
ax.set_xticklabels(folds_x, fontsize=8)
ax.set_ylabel("RMSE", color="salmon")
ax_r.set_ylabel("R²", color="mediumseagreen")
ax.set_title("Per-Fold Metrics")
ax.legend(loc="upper left")
ax_r.legend(loc="upper right")

# 6c. Feature Importance
ax = axes[1, 0]
sorted_idx = np.argsort(importances)
ax.barh(np.array(FEATURE_COLS)[sorted_idx], importances[sorted_idx],
        color="teal", edgecolor="white")
ax.set_xlabel("Importance (MDI)")
ax.set_title("Feature Importance")

# 6d. Residual Distribution
ax = axes[1, 1]
residuals = all_yt - all_yp
ax.hist(residuals, bins=60, color="slategray", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", ls="--", lw=1.5)
ax.set_xlabel("Residual (Observed - Predicted)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
ax.text(0.95, 0.92, f"mean = {residuals.mean():.4f}\nstd  = {residuals.std():.4f}",
        transform=ax.transAxes, fontsize=10, ha="right", verticalalignment="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.95])
PLOT_PATH = "/home/claude/rf_cv_results.png"
plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
print(f"  Plot saved -> {PLOT_PATH}")

print("\n" + "=" * 70)
print("  ✓  ALL DONE")
print("=" * 70)