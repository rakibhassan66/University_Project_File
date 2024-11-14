# analysis.py

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("CDC.csv")  # Replace with the actual filename if different

# Check if data is loaded correctly
print(df.head())  # Display the first few rows
print(df.info())  # Get a summary of columns and data types

# Select only numeric columns for IQR calculation and outlier detection
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the IQR for each numeric column
Q1 = numeric_df.quantile(0.25)  # 25th percentile (lower quartile)
Q3 = numeric_df.quantile(0.75)  # 75th percentile (upper quartile)
IQR = Q3 - Q1                    # Calculate the IQR

# Identify outliers by checking if data points fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
print("Outliers detected:\n", outliers.sum())  # Sum of True values in each column shows the number of outliers

# Optional: Box plot for visualization of numeric columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric_df)  # Only numeric columns
plt.title("Boxplot for Outlier Detection in Numeric Columns")
plt.show()




# With Visual View