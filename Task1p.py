import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('CDC.csv')

# Check the data types of each column
print("Data types before conversion:")
print(df.dtypes)

# Convert all columns to numeric, errors='coerce' turns non-numeric entries into NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Check for NaN values after conversion
print("\nData after conversion (with NaN for non-numeric values):")
print(df.head())

# Fill NaN values with the mean of each column
df = df.fillna(df.mean())

# IQR Method for Outlier Detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Defining the range for detecting outliers (lower and upper bounds)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifying the outliers using IQR
outliers_iqr = ((df < lower_bound) | (df > upper_bound)).sum()
print("\nOutliers detected using IQR method:")
print(outliers_iqr)

# Z-Score Method for Outlier Detection
# Calculate the Z-scores for the DataFrame
z_scores = np.abs(stats.zscore(df))

# Identifying the outliers with Z-score greater than 3
outliers_zscore = (z_scores > 3).sum()
print("\nOutliers detected using Z-score method:")
print(outliers_zscore)

# Optionally, if you want to remove outliers based on IQR or Z-Score:
# Remove outliers using IQR method (values within bounds)
df_no_outliers_iqr = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Remove outliers using Z-score method (values with Z-score > 3)
df_no_outliers_zscore = df[(z_scores <= 3).all(axis=1)]

print("\nData after removing outliers based on IQR method:")
print(df_no_outliers_iqr.head())

print("\nData after removing outliers based on Z-score method:")
print(df_no_outliers_zscore.head())
