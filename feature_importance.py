import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data (used during model training)
df = pd.read_csv("data/processed_credit_data.csv")  # replace with your actual file
X_train = df.drop("Risk", axis=1)  # assuming 'Risk' is the target column

# Load model
model = joblib.load("models/xgb_model.pkl")

# Feature importance
importances = model.feature_importances_
features = X_train.columns

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

# Create a folder if it doesn't exist
import os
output_path = "images"
os.makedirs(output_path, exist_ok=True)

# Save the image inside the folder
image_file = os.path.join(output_path, "feature_importance.png")
plt.savefig(image_file)
print(f"Image saved at: {image_file}")  # Debug message

# Close the plot
plt.close()

#   streamlit run app/streamlit_app.py
