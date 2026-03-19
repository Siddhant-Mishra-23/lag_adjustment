import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------
# 1. LOAD JSON
# -----------------------------
# with open("output/Json/all_results.json", "r") as f:
#     data = json.load(f)
with open("output/Json/CatBoost_error.json", "r") as f:
    data = json.load(f)


# Support both single + multiple models
if isinstance(data, dict):
    data = [data]

# -----------------------------
# 2. FLATTEN DATA + ADD AVERAGE
# -----------------------------
rows = []

for entry in data:
    model = entry["model"]
    cv = entry["cv_type"]

    # Fold rows
    for fold in entry["folds"]:
        rows.append({
            "Model": model,
            "CV Type": cv,
            "Fold": fold["fold"],
            "RMSE": fold["rmse"],
            "MAE": fold["mae"],
            "R2": fold["r2"],
            "UBRMSE": fold["ubrmse"],
            "Bias": fold["bias"]
        })

    # Average row
    avg = entry["average"]
    rows.append({
        "Model": model,
        "CV Type": cv,
        "Fold": "Avg",
        "RMSE": avg["rmse"],
        "MAE": avg["mae"],
        "R2": avg["r2"],
        "UBRMSE": avg["ubrmse"],
        "Bias": avg["bias"]
    })

df = pd.DataFrame(rows)

# -----------------------------
# 3. SORT (important for grouping)
# -----------------------------
df = df.sort_values(["Model", "CV Type", "Fold"], key=lambda x: x.map(lambda v: 999 if v == "Avg" else v)).reset_index(drop=True)

# -----------------------------
# 4. MERGE CELLS (visual trick)
# -----------------------------
for col in ["Model", "CV Type"]:
    df.loc[df[col].duplicated(), col] = ""

# -----------------------------
# 5. NORMALIZATION (for coloring)
# -----------------------------
def normalize(col, reverse=False):
    col = col.astype(float)
    norm = (col - col.min()) / (col.max() - col.min() + 1e-9)
    return 1 - norm if reverse else norm

df_norm = pd.DataFrame()
df_norm["RMSE"] = normalize(df["RMSE"])
df_norm["MAE"] = normalize(df["MAE"])
df_norm["UBRMSE"] = normalize(df["UBRMSE"])
df_norm["Bias"] = normalize(abs(df["Bias"]))
df_norm["R2"] = normalize(df["R2"], reverse=True)

# -----------------------------
# 6. COLOR MAP (smooth gradient)
# -----------------------------
cmap = plt.cm.RdYlGn_r
norm_obj = mcolors.Normalize(vmin=0, vmax=1)

def get_color(val):
    return cmap(norm_obj(val))

# -----------------------------
# 7. BUILD CELL COLORS
# -----------------------------
cell_colors = []

for i in range(len(df)):
    row_colors = []
    for col in df.columns:
        if col in df_norm.columns:
            row_colors.append(get_color(df_norm.loc[i, col]))
        else:
            row_colors.append((1, 1, 1, 1))
    cell_colors.append(row_colors)

# -----------------------------
# 8. FORMAT DISPLAY (FIXED ERROR)
# -----------------------------
df_display = df.copy()

# Format metric columns safely, even when pandas stores them as strings
metric_cols = ["RMSE", "MAE", "R2", "UBRMSE", "Bias"]
for col in metric_cols:
    numeric_vals = pd.to_numeric(df_display[col], errors="coerce")
    df_display[col] = numeric_vals.map(lambda x: f"{x:.5f}" if pd.notna(x) else "")

cell_text = df_display.values

# -----------------------------
# 9. PLOT TABLE
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

table = ax.table(
    cellText=cell_text,
    colLabels=df.columns,
    cellColours=cell_colors,
    loc='center'
)

# -----------------------------
# 10. STYLING
# -----------------------------
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for (row, col), cell in table.get_celld().items():
    # Header styling
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#EAEAEA')

    # Highlight Average rows
    if row > 0 and df.iloc[row - 1]["Fold"] == "Avg":
        cell.set_text_props(weight='bold')
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)

# -----------------------------
# 11. SAVE OUTPUT
# -----------------------------
plt.savefig("output/cv_results_test_CatBoost_error.png", bbox_inches='tight', dpi=300)
plt.close()

excel_output_path = "output/cv_results_final.xlsx"
df_display.to_excel(excel_output_path, index=False)

print("Saved: cv_results_final.png")
print(f"Saved: {excel_output_path}")
