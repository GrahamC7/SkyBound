import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained models
rf_model_path = "../models/rocket_launch_model.pkl"
xgb_model_path = "../models/xgboost_model.pkl"

print("📂 Loading trained models...")
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# Load dataset to get feature names
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)
feature_names = df.drop(columns=["launch_status"]).columns  # Drop target variable

# Extract feature importances
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Debugging: Print feature counts
print(f"🔍 Random Forest Features: {len(rf_importance)}")
print(f"🔍 XGBoost Features: {len(xgb_importance)}")
print(f"🔍 Feature Names Count: {len(feature_names)}")

# Ensure feature lengths match by truncating to the smallest feature set
min_features = min(len(rf_importance), len(xgb_importance), len(feature_names))
rf_importance = rf_importance[:min_features]
xgb_importance = xgb_importance[:min_features]
feature_names = feature_names[:min_features]

# Create a DataFrame for comparison
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "RandomForest": rf_importance,
    "XGBoost": xgb_importance
}).sort_values(by="XGBoost", ascending=False)

# Save the importance comparison as a CSV
output_csv = "../data/feature_importance_comparison.csv"
importance_df.to_csv(output_csv, index=False)
print(f"✅ Feature importance comparison saved as '{output_csv}'")

# Plot feature importance
plt.figure(figsize=(10, 6))
importance_df.plot(x="Feature", y=["RandomForest", "XGBoost"], kind="bar", figsize=(12, 6))
plt.title("Feature Importance Comparison: Random Forest vs XGBoost")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# Save the plot
output_plot = "../visualizations/feature_importance_comparison.png"
plt.savefig(output_plot, dpi=300)
print(f"✅ Feature importance plot saved as '{output_plot}'")

# Show plot
plt.show()
