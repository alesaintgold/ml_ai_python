import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Load the Breast Cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create classifiers
dt_clf = DecisionTreeClassifier(random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42) 
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train classifiers
dt_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Make predictions
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)

# Calculate accuracy
dt_acc = accuracy_score(y_test, dt_pred)
ada_acc = accuracy_score(y_test, ada_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Print performance comparison
print("Decision Tree Accuracy:", dt_acc)
print("AdaBoost Accuracy:", ada_acc)
print("Random Forest Accuracy:", rf_acc)

# Plot learning curve (example with AdaBoost)
train_sizes, train_scores, test_scores = learning_curve(
    ada_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend()
plt.title("AdaBoost Learning Curve")
plt.show()

# Analysis of boosting effect
# AdaBoost combines multiple weak learners (decision trees in this case) 
# to create a strong learner. 
# By iteratively adjusting the weights of misclassified samples, 
# AdaBoost focuses on the most difficult examples, 
# improving the overall model performance.