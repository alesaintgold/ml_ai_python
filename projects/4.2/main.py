import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

print("Loading dataset and splitting in training and testing")
# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Defining hyperparameter grid for RandomizedSearchCV")
param_distributions = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("Creating  a Random Forest Regressor and a RandomizedSearchCV")
# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_distributions, 
                                   n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

print("Training the model")
# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", random_search.best_params_)

# Evaluate the best model on the test set
y_pred = random_search.best_estimator_.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Create a table of results
results_table = {
    'Metric': ['Mean Squared Error', 'R2 Score'],
    'Value': [mse, r2]
}

print(pd.DataFrame(results_table))