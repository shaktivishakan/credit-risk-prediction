import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Load the dataset
df = pd.read_csv(os.path.join('data','german_credit_data.csv'), index_col=0)

print(df['Risk'].value_counts())