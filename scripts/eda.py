import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Load the dataset

df = pd.read_csv(os.path.join('data','german_credit_data.csv'), index_col=0)


# # Basic info

# print(df.head())
# print(df.info())
# print(df.describe())

#check for missing values

# print("\n Missisng Values: /n", df.isnull().sum())

# Plot numeric distribution

numerical_cols = df.select_dtypes(include=['int64','float64']).columns

# for col in numerical_cols:
#     plt.figure(figsize = (6,4))
#     sns.histplot(df[col],kde = True)
#     plt.title(f'Distribution of {col}')
#     plt.tight_layout()
#     plt.savefig(os.path.join('scripts', 'plots', f'{col}_dist.png'))
#     plt.close()

# Categorical Value counts

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f'\n{col} Value counts:')
    print(df[col].value_counts())
