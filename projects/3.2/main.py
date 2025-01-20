import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def bagging_classifier(X_train, y_train, n_trees=10):
  """
  Implements bagging for classification with decision trees as base learners.

  Args:
    X_train: Training data features.
    y_train: Training data labels.
    n_trees: Number of decision trees in the ensemble.

  Returns:
    A list of trained decision trees.
  """

  trees = []
  for _ in range(n_trees):
    # Create bootstrap samples
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bootstrap = X_train[indices]
    y_bootstrap = y_train[indices]

    # Train a decision tree on the bootstrap sample
    tree = DecisionTreeClassifier()
    tree.fit(X_bootstrap, y_bootstrap)
    trees.append(tree)

  return trees

def predict(X_test, trees):
  """
  Predicts the class labels for the given data using the bagging ensemble.

  Args:
    X_test: Test data features.
    trees: List of trained decision trees.

  Returns:
    Predicted class labels.
  """

  predictions = np.zeros((len(X_test), len(trees)))
  for i, tree in enumerate(trees):
    predictions[:, i] = tree.predict(X_test)

  # Ensure predictions are integers before majority voting
  predictions = predictions.astype(int) 

  return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)

def visualize_boundaries(X_train, y_train, X_test, y_test, tree, bagging_trees):
  """
  Visualizes the decision boundaries of a single decision tree and the bagging ensemble.

  Args:
    X_train: Training data features.
    y_train: Training data labels.
    X_test: Test data features.
    y_test: Test data labels.
    tree: A single decision tree.
    bagging_trees: List of trained decision trees for bagging.
  """

  # Create a mesh to plot in
  x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
  y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

  # Predict on the mesh for single tree
  Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()])
  Z_tree = Z_tree.reshape(xx.shape)

  # Predict on the mesh for bagging
  Z_bagging = predict(np.c_[xx.ravel(), yy.ravel()], bagging_trees)
  Z_bagging = Z_bagging.reshape(xx.shape)

  # Plot the decision boundaries
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.contourf(xx, yy, Z_tree, alpha=0.8)
  plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
  plt.title("Single Decision Tree")

  plt.subplot(1, 2, 2)
  plt.contourf(xx, yy, Z_bagging, alpha=0.8)
  plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
  plt.title("Bagging Ensemble")

  plt.show()

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data[:, :2]  # Use only the first two features for visualization
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a single decision tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Create a bagging ensemble
bagging_trees = bagging_classifier(X_train, y_train, n_trees=50)

# Make predictions
y_pred_tree = tree.predict(X_test)
y_pred_bagging = predict(X_test, bagging_trees)

# Calculate accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print(f"Accuracy of single decision tree: {accuracy_tree:.3f}")
print(f"Accuracy of bagging ensemble: {accuracy_bagging:.3f}")

# Visualize decision boundaries
visualize_boundaries(X_train, y_train, X_test, y_test, tree, bagging_trees)