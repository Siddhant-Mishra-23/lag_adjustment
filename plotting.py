import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import gaussian_kde
from scipy.stats import skew

def calculate_metrics(y_actual, y_predicted):
    r2 = r2_score(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))
    mae = mean_absolute_error(y_actual,y_predicted)
    nse = 1 - np.sum((y_actual - y_predicted) ** 2) / np.sum((y_actual -np.mean(y_actual)) ** 2)
    return r2, rmse, mae, nse

def get_model_label(df):
    models = pd.unique(df["model"].dropna())
    return ", ".join(str(model) for model in models)

def _fast_density(x, y, bins=60):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0:
        return np.array([], dtype=float)
    # Edge case: constant values
    if np.isclose(x.min(), x.max()) or np.isclose(y.min(), y.max()):
        return np.ones_like(x, dtype=float)
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
    return hist[x_bin, y_bin]

def plot_regression_scatter(dfs, actual, predicted, save_path, bins=60):
    fig, axes = plt.subplots(2, len(dfs)//2, figsize=(5 * len(dfs), 20))
    axes = axes.flatten()
    if len(dfs) == 1:
        axes = [axes]
    for i, df in enumerate(dfs):
        ax = axes[i]
        y_actual = df[actual].to_numpy()
        y_predicted = df[predicted].to_numpy()
        # ---- Metrics ----
        r2, rmse, mae, nse = calculate_metrics(y_actual, y_predicted)
        model_label = get_model_label(df)
        # ---- FAST density ----
        density = _fast_density(y_actual, y_predicted, bins=bins)
        order = np.argsort(density)
        x = y_actual[order]
        y = y_predicted[order]
        d = density[order]
        sc = ax.scatter(x, y,c=d,cmap="jet",s=8,alpha=0.8,edgecolors="none")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_title(model_label ,fontsize= 24)
        ax.set_xlabel("Actual Soil Moisture (m³/m³)",fontsize= 21)
        ax.set_ylabel("Predicted Soil Moisture (m³/m³)",fontsize= 21)
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        text = "\n".join([
            f"R² = {r2:.3f}",
            f"RMSE = {rmse:.3f}",
            f"MAE = {mae:.3f}",
            f"NSE = {nse:.3f}"
        ])

        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=21,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.85)
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()    

  
def plot_regression_skewness(dfs, actual, predicted, save_path):
    fig, axes = plt.subplots(2, len(dfs)//2, figsize=(20, 20))
    axes = axes.flatten()
    for i, df in enumerate(dfs):
        y_actual = df[actual].to_numpy()
        y_predicted = df[predicted].to_numpy()
        residual = y_actual - y_predicted
        model_label = get_model_label(df)
        ax1 = axes[i]
        ax1.hist(residual, bins=50, density=True, alpha=0.6, edgecolor="black", linewidth=0.8, color="darkblue")
        ax1.set_xlim(-0.1, 0.1)
        try:
            kde = gaussian_kde(residual)
            x_vals = np.linspace(residual.min(), residual.max(), 200)
            ax1.plot(x_vals, kde(x_vals),color='red',linewidth=6)
        except:
            pass
        skewness = skew(residual)
        ax1.set_title(f"Residual Distribution for: {model_label}", fontsize= 24)
        ax1.set_xlabel("Residual (Actual - Predicted)",fontsize= 21)
        ax1.set_ylabel("Density",fontsize= 21)
        ax1.tick_params(axis='both', labelsize=14)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
        text = (
            f"Skewness = {skewness:.3f}/n"
        )
        ax1.text(0.025, 0.95, text, transform=ax1.transAxes,
                 verticalalignment='top', fontsize=24)
                 #,bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    RF_GFK_TRAINING = "output/Training Metadata/RandomForest_groupkfold_train.parquet"
    CB_GFK_TRAINING = "output/Training Metadata/CatBoost_groupkfold_train.parquet"
    XGB_GFK_TRAINING = "output/Training Metadata/XGBoost_groupkfold_train.parquet"
    LGBM_GFK_TRAINING = "output/Training Metadata/LIGHTGBM_groupkfold_train.parquet"
    df_rf_gfk_training = pd.read_parquet(RF_GFK_TRAINING)
    df_cb_gfk_training = pd.read_parquet(CB_GFK_TRAINING)
    df_xgb_gfk_training = pd.read_parquet(XGB_GFK_TRAINING)
    df_lgbm_gfk_training = pd.read_parquet(LGBM_GFK_TRAINING)
    print("Executing for the printing skewness plot for GFK Training Data")
    plot_regression_skewness(
        dfs = [ df_rf_gfk_training, df_cb_gfk_training, df_xgb_gfk_training, df_lgbm_gfk_training] ,
        actual= "y_train",
        predicted= "y_pred_train",
        save_path= "output/plots/skewness_regression_gfk.png"
        )
    print("Printing skewness plot for GFK Training Data completed")
    print("Executing for the printing scatter plot for GFK Training Data")
    plot_regression_scatter(
        dfs = [ df_rf_gfk_training, df_cb_gfk_training, df_xgb_gfk_training, df_lgbm_gfk_training] ,
        actual= "y_train",
        predicted= "y_pred_train",
        save_path= "output/plots/scatter_regression_gfk.png"
        )
    print("Printing scatter plot for GFK Training Data completed")
    del df_cb_gfk_training, df_rf_gfk_training, df_lgbm_gfk_training, df_xgb_gfk_training

    RF_GFK_VALIDATION = "output/Training Metadata/RandomForest_groupkfold_valid.parquet"
    CB_GFK_VALIDATION = "output/Training Metadata/CatBoost_groupkfold_valid.parquet"
    XGB_GFK_VALIDATION = "output/Training Metadata/XGBoost_groupkfold_valid.parquet"
    LGBM_GFK_VALIDATION ="output/Training Metadata/LIGHTGBM_groupkfold_valid.parquet"
    df_rf_gfk_validation = pd.read_parquet(RF_GFK_VALIDATION)
    df_cb_gfk_validation = pd.read_parquet(CB_GFK_VALIDATION)
    df_xgb_gfk_validation = pd.read_parquet(XGB_GFK_VALIDATION)
    df_lgbm_gfk_validation = pd.read_parquet(LGBM_GFK_VALIDATION)
    print("Executing for the printing scatter plot for GFK Validation Data")
    plot_regression_scatter(
        dfs = [ df_rf_gfk_validation, df_cb_gfk_validation, df_xgb_gfk_validation, df_lgbm_gfk_validation] ,
        actual= "y_test",
        predicted= "y_pred_test",
        save_path= "output/plots/scatter_regression_gfk_validation.png"
        )
    print("Printing scatter plot for GFK Validation Data completed")
    print("Executing for the printing skewness plot for GFK Validation Data")
    plot_regression_skewness(
        dfs = [ df_rf_gfk_validation, df_cb_gfk_validation, df_xgb_gfk_validation, df_lgbm_gfk_validation] ,
        actual= "y_test",
        predicted= "y_pred_test",
        save_path= "output/plots/skewness_regression_gfk_validation.png"
        )
    print("Printing skewness plot for GFK Validation Data completed")
    del df_cb_gfk_validation, df_rf_gfk_validation, df_lgbm_gfk_validation, df_xgb_gfk_validation

    RF_TS_TRAINING = "output/Training Metadata/RandomForest_timeseries_year_train.parquet"
    CB_TS_TRAINING = "output/Training Metadata/CatBoost_timeseries_year_train.parquet"
    XGB_TS_TRAINING = "output/Training Metadata/XGBoost_timeseries_year_train.parquet"
    LGBM_TS_TRAINING = "output/Training Metadata/LIGHTGBM_timeseries_year_train.parquet"
    df_rf_ts_training = pd.read_parquet(RF_TS_TRAINING)
    df_cb_ts_training = pd.read_parquet(CB_TS_TRAINING)
    df_xgb_ts_training = pd.read_parquet(XGB_TS_TRAINING)
    df_lgbm_ts_training = pd.read_parquet(LGBM_TS_TRAINING)
    print("Executing for the printing scatter plot for TS Training Data")
    plot_regression_scatter(
        dfs = [ df_rf_ts_training, df_cb_ts_training, df_xgb_ts_training, df_lgbm_ts_training] ,
        actual= "y_train",
        predicted= "y_pred_train",
        save_path= "output/plots/scatter_regression_ts.png"
        )
    print("Printing scatter plot for TS Training Data completed")
    print("Executing for the printing skewness plot for TS Training Data")
    plot_regression_skewness(
        dfs = [ df_rf_ts_training, df_cb_ts_training, df_xgb_ts_training, df_lgbm_ts_training] ,
        actual= "y_train",
        predicted= "y_pred_train",
        save_path= "output/plots/skewness_regression_ts.png"
        )
    print("Printing skewness plot for TS Training Data completed")
    del df_cb_ts_training, df_rf_ts_training, df_lgbm_ts_training, df_xgb_ts_training


    RF_TS_VALIDATION = "output/Training Metadata/RandomForest_timeseries_year_valid.parquet"
    CB_TS_VALIDATION = "output/Training Metadata/CatBoost_timeseries_year_valid.parquet"
    XGB_TS_VALIDATION = "output/Training Metadata/XGBoost_timeseries_year_valid.parquet"
    LGBM_TS_VALIDATION = "output/Training Metadata/LIGHTGBM_timeseries_year_valid.parquet"
    df_rf_ts_validation = pd.read_parquet(RF_TS_VALIDATION)
    df_cb_ts_validation = pd.read_parquet(CB_TS_VALIDATION)
    df_xgb_ts_validation = pd.read_parquet(XGB_TS_VALIDATION)
    df_lgbm_ts_validation = pd.read_parquet(LGBM_TS_VALIDATION)
    print("Executing for the printing scatter plot for TS Validation Data")
    plot_regression_scatter(
        dfs = [ df_rf_ts_validation, df_cb_ts_validation, df_xgb_ts_validation, df_lgbm_ts_validation] ,
        actual= "y_test",
        predicted= "y_pred_test",
        save_path= "output/plots/scatter_regression_ts_validation.png"
        )
    print("Printing scatter plot for TS Validation Data completed")
    print("Executing for the printing skewness plot for TS Validation Data")
    plot_regression_skewness(
        dfs = [ df_rf_ts_validation, df_cb_ts_validation, df_xgb_ts_validation, df_lgbm_ts_validation] ,
        actual= "y_test",
        predicted= "y_pred_test",
        save_path= "output/plots/skewness_regression_ts_validation.png"
        )
    print("Printing skewness plot for TS Validation Data completed")
    del df_cb_ts_validation, df_rf_ts_validation, df_lgbm_ts_validation, df_xgb_ts_validation



    RF_TESTING = "output/Testing Metadata/RandomForest_sm_pred.parquet"
    CB_TESTING = "output/Testing Metadata/CatBoost_sm_pred.parquet"
    XGB_TESTING = "output/Testing Metadata/XGBoost_sm_pred.parquet"
    LGBM_TESTING = "output/Testing Metadata/LIGHTGBM_sm_pred.parquet"
    df_rf_testing = pd.read_parquet(RF_TESTING)
    df_rf_testing["model"] = "RandomForest"
    df_cb_testing = pd.read_parquet(CB_TESTING)
    df_cb_testing["model"] = "CatBoost"
    df_xgb_testing = pd.read_parquet(XGB_TESTING)
    df_xgb_testing["model"] = "XGBoost"
    df_lgbm_testing = pd.read_parquet(LGBM_TESTING)
    df_lgbm_testing["model"] = "LightGBM"
    print("Executing for the printing scatter plot for test Data")
    plot_regression_scatter(
        dfs = [ df_rf_testing, df_cb_testing, df_xgb_testing, df_lgbm_testing] ,
        actual= "sm",
        predicted= "sm_pred",
        save_path= "output/plots/scatter_regression_test.png"
        )
    print("Printing scatter plot for TS Validation Data completed")
    print("Executing for the printing skewness plot for TS Validation Data")
    plot_regression_skewness(
        dfs = [ df_rf_testing, df_cb_testing, df_xgb_testing, df_lgbm_testing] ,
        actual= "sm",
        predicted= "sm_pred",
        save_path= "output/plots/skewness_regression_test.png"
        )
    print("Printing skewness plot for TS Validation Data completed")
    del df_rf_testing, df_cb_testing, df_xgb_testing, df_lgbm_testing




    







