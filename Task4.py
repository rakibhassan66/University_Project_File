import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
data = pd.read_csv("CDC.csv")

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

# Calculate prior probabilities for each class
class_priors = y_train.value_counts(normalize=True).to_dict()

# Calculate mean and variance for each feature in each class (for Gaussian Naive Bayes)
feature_stats = {}
for owner_class in y_train.unique():
    X_class = X_train[y_train == owner_class]
    feature_stats[owner_class] = {
        'mean': X_class.mean(),
        'variance': X_class.var()
    }

# Function to calculate Gaussian likelihood
def gaussian_likelihood(x, mean, var):
    eps = 1e-6  # to avoid division by zero
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(-(x - mean) ** 2 / (2 * var + eps))
    return coeff * exponent

# Calculate posterior probabilities for each test instance
posterior_probs = []
for index, row in X_test.iterrows():
    posteriors = {}
    for owner_class, stats in feature_stats.items():
        prior = class_priors[owner_class]
        likelihood = prior
        for feature in X_test.columns:
            mean = stats['mean'][feature]
            var = stats['variance'][feature]
            likelihood *= gaussian_likelihood(row[feature], mean, var)
        posteriors[owner_class] = likelihood
    posterior_probs.append(posteriors)

# Print the posterior probabilities for the test instances
for i, probs in enumerate(posterior_probs):
    print(f"Test instance {i}: Posterior probabilities -> {probs}")
