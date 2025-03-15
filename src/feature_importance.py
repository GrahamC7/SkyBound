import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model_path = "../models/rocket_launch_model.pkl"
model = joblib.load(model_path)

# Load preprocessed data to get feature names
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Drop non-numeric and target columns
non_numeric_columns = ["date", "launch_location", "Status Mission", "time", "launch_status"]
df = df.drop(columns=non_numeric_columns, errors="ignore")

# Extract feature importance
feature_importance = model.feature_importances_
feature_names = df.columns

# Create DataFrame for plotting
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Rocket Launch Prediction")
plt.show()
