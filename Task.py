# task.py
import pandas as pd

# Load the dataset
df = pd.read_csv("CDC.csv")

# Display basic info
print("Dataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())
