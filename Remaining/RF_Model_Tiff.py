import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import joblib

# === Load and combine Excel sheets ===
print("📥 Loading and combining all sheets from Excel file...")

excel_path = "Filtered_Data_2020-2024.xlsx"
xlsx = pd.ExcelFile(excel_path)

# ✅ This line loads all sheets
df = pd.concat([xlsx.parse(sheet) for sheet in tqdm(xlsx.sheet_names)], ignore_index=True)

print(f"✅ Combined DataFrame shape: {df.shape}")

# === Preprocessing ===
print("🧹 Preprocessing data...")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Latitude', 'Longitude', 'Date'])

# Create prediction target: SMM_t+3
df['SMM_t+3'] = df.groupby(['Latitude', 'Longitude'])['SMM'].shift(-3)
df = df.dropna(subset=['SMM_t+3'])

# === Feature-target setup ===
features = ['Rainfall_mm', 'Temperature', 'SMM']
target = 'SMM_t+3'

X = df[features]
y = df[target]

# === Train-Test Split ===
print("🔄 Splitting into training and testing sets...")
split_index = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# === Model Training ===
print("🌲 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Prediction and Evaluation ===
print("📈 Evaluating model performance...")
y_pred = model.predict(X_test)

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
nse = r2  # For hydrological models, NSE ≈ R²
smape = np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100
mhe = np.mean(y_pred - y_test)

# === Results ===
print("\n✅ Evaluation Results:")
print(f"🔹 RMSE  : {rmse:.4f}")
print(f"🔹 MAE   : {mae:.4f}")
print(f"🔹 R²    : {r2:.4f}")
print(f"🔹 NSE   : {nse:.4f}")
print(f"🔹 SMAPE : {smape:.2f}%")
print(f"🔹 MHE   : {mhe:.4f}")

# === Save model to pickle ===
joblib.dump(model, "random_forest_smm_model_filtered_3days_Tiff.pkl")
print("✅ Model saved to: random_forest_smm_model_filtered_3days_Tiff.pkl")


