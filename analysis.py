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