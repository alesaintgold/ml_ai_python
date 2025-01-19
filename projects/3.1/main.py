import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the Boston Housing dataset
data = fetch_california_housing()
X = data['data']
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of base models in the ensemble
num_models = 50

# Create a list to store the predictions of each base model
predictions = []

# Train multiple regression models on bootstrapped samples
for i in range(num_models):
    # Create a bootstrap sample
    bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_train_bootstrap = X_train[bootstrap_indices]
    y_train_bootstrap = y_train[bootstrap_indices]

    # Train a regression model on the bootstrap sample
    model = LinearRegression()
    model.fit(X_train_bootstrap, y_train_bootstrap)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    predictions.append(y_pred)

# Ensemble predictions by averaging the predictions of all base models
ensemble_predictions = np.mean(predictions, axis=0)

# Calculate MSE for single regression model
single_model = LinearRegression()
single_model.fit(X_train, y_train)
single_model_pred = single_model.predict(X_test)
single_model_mse = mean_squared_error(y_test, single_model_pred)

# Calculate MSE for bagging ensemble
bagging_mse = mean_squared_error(y_test, ensemble_predictions)

# Print MSE comparison
print("Single Model MSE:", single_model_mse)
print("Bagging Ensemble MSE:", bagging_mse)

# Create a table for comparison
comparison_table = {
    "Model": ["Single Regression", "Bagging Ensemble"],
    "MSE": [single_model_mse, bagging_mse]
}
print("\nMSE Comparison:")
for row in zip(*comparison_table.values()):
    print("{: <20} {:>10}".format(*row))

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, single_model_pred, label="Single Model")
plt.scatter(y_test, ensemble_predictions, label="Bagging Ensemble")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Bagging Ensemble vs. Single Regression")
plt.show()