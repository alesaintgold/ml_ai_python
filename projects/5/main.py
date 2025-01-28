import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize predictions with the mean of the target variable
        y_pred = np.mean(y) * np.ones_like(y)

        for i in range(self.n_estimators):
            # Calculate residuals
            residuals = y - y_pred

            # Train a decision tree on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

            # Store the trained tree
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Load the Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)

# Train Linear Regression model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)

# Predict on test data
y_pred_gb = gb_regressor.predict(X_test)
y_pred_lr = lr_regressor.predict(X_test)

# Calculate Mean Squared Error
mse_gb = mean_squared_error(y_test, y_pred_gb)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Gradient Boosting MSE: {mse_gb}")
print(f"Linear Regression MSE: {mse_lr}")

# Discussion of model behavior
if mse_gb < mse_lr:
    print("Gradient Boosting generally performs better than Linear Regression, "
          "especially when the data is non-linear or has complex relationships.")
else:
    print("Linear Regression might be a better choice for this dataset.")

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_gb, y_test - y_pred_gb, label="Gradient Boosting")
plt.scatter(y_pred_lr, y_test - y_pred_lr, label="Linear Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.title("Residual Plots")
plt.show()
