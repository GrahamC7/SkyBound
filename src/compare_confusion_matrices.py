import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 📂 Load preprocessed data
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# 📂 Load trained models
rf_model_path = "../models/rocket_launch_rf_retrained.pkl"
xgb_model_path = "../models/xgboost_model_retrained.pkl"
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# Define target column
target_column = "launch_status"

# 🔄 Drop non-numeric columns
non_numeric_columns = ["date", "launch_location", "Status Mission", "time"]
df = df.drop(columns=non_numeric_columns, errors="ignore")

# 🚀 Ensure we **only use the features that were present in training**
all_features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m',
                'windgusts_10m', 'winddirection_10m', 'cloudcover', 'pressure_msl']

# ✅ Drop any extra columns that may have been introduced
X = df[all_features]  # **Keep only the features used in training**
y = df[target_column]

# ✅ Ensure labels are binary (0,1)
assert set(np.unique(y)) == {0, 1}, f"Unexpected values in y: {np.unique(y)}"

# 🔄 Define thresholds to test
thresholds = [0.45, 0.50, 0.55]
conf_matrices_xgb = {}
conf_matrices_rf = {}

for threshold in thresholds:
    # 🔥 Make predictions for XGBoost
    y_pred_xgb = (xgb_model.predict_proba(X)[:, 1] >= threshold).astype(int)
    cm_xgb = confusion_matrix(y, y_pred_xgb)
    conf_matrices_xgb[threshold] = cm_xgb

    # 🌲 Make predictions for Random Forest
    y_pred_rf = (rf_model.predict_proba(X)[:, 1] >= threshold).astype(int)
    cm_rf = confusion_matrix(y, y_pred_rf)
    conf_matrices_rf[threshold] = cm_rf

    print(f"\n🔄 Confusion Matrices (Threshold = {threshold}):")
    print(f"XGBoost:\n{cm_xgb}")
    print(f"Random Forest:\n{cm_rf}")

# 📊 Plot Confusion Matrices for Different Thresholds
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, threshold in enumerate(thresholds):
    # XGBoost
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=conf_matrices_xgb[threshold], display_labels=["No-Go", "Go"])
    disp_xgb.plot(ax=axes[0, idx], cmap="Blues", values_format="d")
    axes[0, idx].set_title(f"XGBoost (Threshold = {threshold})")

    # Random Forest
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrices_rf[threshold], display_labels=["No-Go", "Go"])
    disp_rf.plot(ax=axes[1, idx], cmap="Blues", values_format="d")
    axes[1, idx].set_title(f"Random Forest (Threshold = {threshold})")

plt.tight_layout()

# 💾 Save the figure
output_path = "../visualizations/threshold_comparison.png"
plt.savefig(output_path)
print(f"✅ Threshold comparison saved as '{output_path}'")

plt.show()
