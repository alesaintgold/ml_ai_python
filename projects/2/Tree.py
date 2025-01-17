import numpy as np

class TreeNode:
    def __init__(self, feature_idx=None, feature_val=None, left = None, right = None):
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.left = left
        self.right = right

class myDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_leaf_nodes=None, min_impurity_decrease=0.0, impurity_index=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.n_leaves = 0
        self.root = None
        if impurity_index in ('gini','entropy'):
            self.impurity_index = impurity_index
        else:
            self.impurity_index = None

    def fit(self, X,y):
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.max_leaf_nodes is not None and self.n_leaves >= self.max_leaf_nodes) or \
           n_samples < self.min_samples_leaf or \
           n_classes == 1:
            self.n_leaves += 1
            return TreeNode(feature_val=self._most_common_label(y))
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # Check if the split provides sufficient impurity decrease
        if best_feature is None or \
           (self.min_impurity_decrease > 0 and
            self._impurity_decrease(y, X[:, best_feature], best_threshold) < self.min_impurity_decrease):
            self.n_leaves += 1
            return TreeNode(feature_val=self._most_common_label(y))
        
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Check if split produces nodes with at least min_samples_leaf
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            self.n_leaves += 1
            return TreeNode(feature_val=self._most_common_label(y))
        
        # Recursively build left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return TreeNode(feature_idx=best_feature, feature_val=best_threshold, left=left, right=right)

    def _best_split(self, X, y):
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

        # Calculate the entropy of the parent
        parent_impurity = self._entropy(y)
        
        # Split the data
        left_indexes, right_indexes = self._split(X, threshold)
        
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0
        
        # Calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_indexes), len(right_indexes)
        e_l, e_r = self._calculate_impurity(y[left_indexes]), self._calculate_impurity(y[right_indexes])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate the information gain
        information_gain = parent_impurity - child_impurity
        return information_gain

    def _split(self, X_column, threshold):
        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()
        return left_indexes, right_indexes
    
    def _impurity_decrease(self, y, X_column, threshold):
        # Calculate impurity decrease
        parent_impurity = self._calculate_impurity(y)
        
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        
        if n_l == 0 or n_r == 0:
            return 0
        
        left_impurity = self._calculate_impurity(y[left_idxs])
        right_impurity = self._calculate_impurity(y[right_idxs])
        
        impurity_decrease = parent_impurity - (n_l / n) * left_impurity - (n_r / n) * right_impurity
        return impurity_decrease

    def _calculate_impurity(self, y):
        #not supporting gini index yet
        return self._entropy(y)
    
    def _gini(self,y):
        pass

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.feature_idx is None:
            return node.feature_val
        
        if x[node.feature_idx] <= node.feature_val:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)