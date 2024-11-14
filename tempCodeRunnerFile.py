import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv("CDC.csv")

# Display sample data
print(data.head())

# Encode categorical features
label_encoders = {}
for column in ['brand', 'fuel', 'owner']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define feature set and target variable
X = data[['brand', 'km_driven', 'fuel', 'selling_price']]
y = data['owner']

# Scale 'selling_price' feature
scaler = StandardScaler()
X['selling_price'] = scaler.fit_transform(X[['selling_price']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred, zero_division=1))
