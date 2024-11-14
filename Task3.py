# task3.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the dataset
df = pd.read_csv("CDC.csv")

# Print column names to identify the correct column
print("Column names in the dataset:")
print(df.columns)

# Choose a numerical column (replace 'price' with the actual column name you want to test)
column_name = 'selling_price'  # Update with your actual numerical column name

# Ensure that the column exists in the DataFrame
if column_name not in df.columns:
    print(f"Error: Column '{column_name}' does not exist in the dataset.")
else:
    # Check for Normal Distribution using Shapiro-Wilk Test
    stat, p_value = stats.shapiro(df[column_name].dropna())  # Drop NaN values if any
    print(f"\nShapiro-Wilk Test for Normality (p-value): {p_value}")

    # Visualize the Histogram for Normal Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name].dropna(), kde=True)
    plt.title(f"Histogram of {column_name} for Normal Distribution")
    plt.show()

    # Q-Q Plot to visually check for normality
    stats.probplot(df[column_name].dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {column_name}")
    plt.show()

    # Binomial Distribution Test
    # Binomial distribution is relevant if your data is binary (0 and 1)
    if df[column_name].nunique() == 2:  # Check if the data is binary
        p_value_binomial = stats.binom_test(df[column_name].sum(), n=len(df[column_name]), p=0.5, alternative='two-sided')
        print(f"\nBinomial Test for {column_name} (p-value): {p_value_binomial}")
    else:
        print(f"\nThe column {column_name} is not binary. Skipping binomial test.")

    # Poisson Distribution Test
    # For Poisson distribution, check if mean ≈ variance
    mean_value = df[column_name].mean()
    var_value = df[column_name].var()
    print(f"\nMean of {column_name}: {mean_value}")
    print(f"Variance of {column_name}: {var_value}")

    # Check if mean ≈ variance for Poisson
    if abs(mean_value - var_value) < 0.1 * mean_value:
        print(f"{column_name} may follow a Poisson distribution.")

    # Perform Poisson Goodness-of-Fit Test
    observed_counts = df[column_name].value_counts().sort_index()
    expected_counts = stats.poisson.pmf(observed_counts.index, mean_value) * len(df[column_name])
    chi_square_stat, p_value_poisson = stats.chisquare(observed_counts, expected_counts)
    print(f"\nPoisson Goodness-of-Fit Test for {column_name} (p-value): {p_value_poisson}")
