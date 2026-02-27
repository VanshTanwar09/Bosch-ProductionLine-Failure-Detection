import pandas as pd
import numpy as np

print("Loading Bosch data...")
train_num = pd.read_csv('train_numeric.csv', index_col=0, nrows=200000)  # Safe size
train_cat = pd.read_csv('train_categorical.csv', index_col=0, nrows=200000)

print("Shapes:", train_num.shape, train_cat.shape)
print("Failures:", train_num['Response'].value_counts(normalize=True))  # ~1.2%

# Quick Clean Numeric
print("\nCleaning numeric...")
X_num = train_num.drop('Response', axis=1).fillna(0)
# Drop zero columns (Bosch standard)
zero_cols = (X_num == 0).mean() > 0.99  # 99% zeros = useless
X_num = X_num.loc[:, ~zero_cols]
print("After zero drop:", X_num.shape)  # ~970 → ~400 cols

# Top varying columns only
varying_cols = X_num.std().nlargest(200).index  # Best 200 sensors
X_clean = X_num[varying_cols]
y = train_num['Response']

print("Final clean shape:", X_clean.shape)  # (200000, 200)
print("Top Station 32 sensor:", varying_cols[varying_cols.str.contains('S32')])

# Save cleaned data
X_clean.to_csv('X_clean.csv')
y.to_csv('y_clean.csv')
print("CLEANED DATA SAVED! Ready for modeling.")
