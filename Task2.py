import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("CDC.csv")

# Filter out non-numeric columns for correlation calculation
df_numeric = df.select_dtypes(include=['number'])

# Calculate Correlation for only numeric columns
correlation_matrix = df_numeric.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize Correlation with Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Apply Simple Linear Regression
# Replace 'X_column' and 'Y_column' with actual column names in your dataset
X = df[['X_column']].values
y = df['Y_column'].values

model = LinearRegression()
model.fit(X, y)

# Predict values using the trained model
y_pred = model.predict(X)

# Calculate accuracy using R-squared and MSE
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

# Plotting Regression Line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X_column')
plt.ylabel('Y_column')
plt.title("Regression Analysis")
plt.legend()
plt.show()
