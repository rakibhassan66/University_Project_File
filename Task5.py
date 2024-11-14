# task5.py
import pandas as pd
import scipy.stats as stats

# Load the dataset
df = pd.read_csv("CDC.csv")

# **One-Sample t-test**
# Here, test if the mean of 'selling_price' is equal to a hypothesized population mean.
# Replace 'selling_price' with the actual column name from your dataset if needed.
# Replace 'population_mean' with the value you want to test against (e.g., 5000).
population_mean = 5000  # Set your hypothesized population mean here
t_stat, p_value = stats.ttest_1samp(df['selling_price'], population_mean)
print("One-sample t-test results:")
print(f"T-statistic: {t_stat}, p-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis (significant difference from the population mean).")
else:
    print("Result: Fail to reject the null hypothesis (no significant difference from the population mean).")

print("\n" + "-"*40 + "\n")

# **Two-Sample t-test**
# Compare the means of two independent columns (e.g., 'age' and 'height') from your dataset.
# Replace 'column_1' and 'column_2' with actual column names.
# Ensure these columns are independent and numerical for a valid two-sample t-test.
t_stat, p_value = stats.ttest_ind(df['km_driven'], df['selling_price'])  # Replace with actual column names
print("Two-sample t-test results:")
print(f"T-statistic: {t_stat}, p-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis (means are significantly different).")
else:
    print("Result: Fail to reject the null hypothesis (no significant difference in means).")

print("\n" + "-"*40 + "\n")

# **Chi-Square Test**
# For testing associations between two categorical variables (e.g., 'gender' and 'smoker_status').
# Replace 'categorical_var1' and 'categorical_var2' with actual categorical column names.
# The chi-square test checks if there is a significant association between the two categories.
contingency_table = pd.crosstab(df['brand'], df['owner'])  # Replace with actual columns
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-square test results:")
print(f"Chi-square Statistic: {chi2_stat}, p-value: {p_value}, Degrees of freedom: {dof}")

# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis (significant association between variables).")
else:
    print("Result: Fail to reject the null hypothesis (no significant association between variables).")

print("\nExpected Frequencies:\n", expected)
