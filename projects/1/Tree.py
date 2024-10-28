import numpy as np

from datetime import datetime

import logging
logging.basicConfig(filename=f"logs/{datetime.now()}.log", level=logging.DEBUG)
logging.info('message')


class TreeNode:
    def __init__(self, feature_idx=None, feature_val=None, left = None, right = None):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.left = left
        self.right = right
\
class DecisionTreeClassifier:
    def __init__(self, max_depth):
        logging.info(f"Initiating classifier: max_depth={max_depth}\n\n")
        self.max_depth = max_depth
        self.root = None

    def fit(self, X,y):
        logging.info(f"Training:\n\tX={X.tobytes()}\n\ty={y.tobytes()}")
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X,y,depth=0, 
        max_depth=5, 
        #min_samples_leaf=10, 
        #max_leaf_nodes=35,
        #min_impurity_decrease=0.05
            ):

        logging.info(f"Growing tree:\n\tX={X.tobytes()}\n\ty={y.tobytes()}\n\tdepth={depth}\n\tmax_depth={max_depth}")
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if max_depth==depth or n_classes == 1 or n_samples < 2:
            leaf_val = self._most_common_label(y)
            return TreeNode(feature_val=leaf_val)

        feature_idx, threshold = self._best_split(X, y)
        
        left_indexes, right_indexes = self._split(X[:, feature_idx], threshold)

        left = self._grow_tree(X[left_indexes, :], y[left_indexes], depth+1)
        right = self._grow_tree(X[right_indexes, :], y[right_indexes], depth+1)

        return TreeNode(feature_idx=feature_idx, feature_val=threshold, 
            left=left, right=right)

    def _most_common_label(self, y):
        logging.info(f"Searching most common label in y={y.tobytes()}")
        return np.bincount(y).argmax()

    def _best_split(self, X, y):
        logging.info(f"Searching most common label in:\n\ty={y.tobytes()}\n\tX={X.tobytes()}")
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _information_gain(self, y, X, threshold):
        logging.info(f"Calculating information gain in:\n\ty={y.tobytes()}\n\tX={X.tobytes()}\n\tthreshold={threshold}")

        # Calculate the entropy of the parent
        parent_entropy = self._entropy(y)
        
        # Split the data
        left_indexes, right_indexes = self._split(X, threshold)
        
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0
        
        # Calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_indexes), len(right_indexes)
        e_l, e_r = self._entropy(y[left_indexes]), self._entropy(y[right_indexes])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, threshold):
        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()
        return left_indexes, right_indexes

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.feature_val is not None:
            return node.feature_val
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)