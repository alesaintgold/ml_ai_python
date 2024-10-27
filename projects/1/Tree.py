import numpy as np

class TreeNode:
    def __init__(self, feature_idx, feature_val, 
    left = None, right = None, prediction = None):
        
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.left = left
        self.right = right
        self.prediction = prediction

    def find_best_split(X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(len(X[1])):
            thresholds = np.unique(X[:][feature])
            for threshold in thresholds:
                gain = self._information_gain(y,x,threshold)
                if gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain
        
        return best_feature, best_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, y, X_column, threshold):

        #Calculate parent entropy
        parent_entropy = entropy(y)
        
        # Split the data
        left_idxs, right_idxs = split(y,X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(y,X_column,threshold):
        left =  []
        right = []
        for row in y:
            if row[X_column]>threshold:
                right.append(row)
            else:
                left.append(row)
        return left, right

    def grow_tree(self, training_data, depth, max_depth=5, 
    min_samples_leaf=10, max_leaf_nodes=35,
    min_impurity_decrease=0.05):

        pass
