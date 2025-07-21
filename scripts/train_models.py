import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import joblib
import os
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(os.path.join('data','german_credit_data.csv'), index_col=0)

print(df['Risk'].value_counts())

# drop irrelevant columns

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Encode Categorical columns

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Define X and y
X = df.drop('Risk', axis=1 )
y = df['Risk']

# train-test split
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# train random forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\nRando, Forest Accuracy", accuracy_score(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# save random forest
joblib.dump(rf, os.path.join('models', 'rf_model.pkl'))

# train xgBoost
xgb = XGBClassifier(scale_pos_weight=300/700)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print("\nXGBoost Accuracy", accuracy_score(y_test, xgb_preds))
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

os.makedirs('images', exist_ok=True)

# Compute confusion matrix
cm = confusion_matrix(y_test, xgb_preds)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()

# Save the image to images folder
plt.savefig(os.path.join('images', 'confusion_matrix.png'))
plt.close()

# save xgboost
joblib.dump(xgb, os.path.join('models', 'xgb_model.pkl'))

# Save preprocessed data for future use (like in feature importance script)
df.to_csv(os.path.join('data', 'processed_credit_data.csv'), index=False)

