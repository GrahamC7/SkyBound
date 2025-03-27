import joblib
import pandas as pd
import matplotlib.pyplot as plt

# load trained xgboost model
xgb_model_path = "../models/xgboost_best_model.pkl"  # use latest model
print("Loading trained XGBoost model...")
xgb_model = joblib.load(xgb_model_path)

# load dataset to get feature names
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)
feature_names = df.drop(columns=["launch_status"]).columns  # drop target variable

# extract feature importances
xgb_importance = xgb_model.feature_importances_

# debugging: print feature counts
print(f"XGBoost Features: {len(xgb_importance)}")
print(f"Feature Names Count: {len(feature_names)}")

# ensure feature lengths match by truncating to the smallest feature set
min_features = min(len(xgb_importance), len(feature_names))
xgb_importance = xgb_importance[:min_features]
feature_names = feature_names[:min_features]

# create a dataframe for comparison
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "XGBoost": xgb_importance
}).sort_values(by="XGBoost", ascending=False)

# save the importance comparison as a csv
output_csv = "../data/feature_importance_comparison.csv"
importance_df.to_csv(output_csv, index=False)
print(f"Feature importance comparison saved as '{output_csv}'")

# plot feature importance
plt.figure(figsize=(10, 6))
importance_df.plot(x="Feature", y=["XGBoost"], kind="bar", figsize=(12, 6))
plt.title("Feature Importance: XGBoost Model")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# save the plot
output_plot = "../visualizations/feature_importance_comparison.png"
plt.savefig(output_plot, dpi=300)
print(f"Feature importance plot saved as '{output_plot}'")

# show plot
plt.show()
