import numpy as np
import pandas as pd

class ID3:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._id3(X, y)

    def _id3(self, X, y):
        # If all labels are the same, return the label
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]

        # If no more features to split on, return the majority label
        if X.shape[1] == 0:
            return np.bincount(y).argmax()

        # Calculate Information Gain for each feature
        best_feature = None
        best_threshold = None
        max_info_gain = -np.inf
        best_left = None
        best_right = None

        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            if np.issubdtype(feature_values.dtype, np.number):  # Continuous data
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    info_gain = self._information_gain(y, left_y, right_y)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_feature = feature_index
                        best_threshold = threshold
                        best_left = left_y
                        best_right = right_y
            else:  # Categorical data
                feature_values = np.unique(feature_values)
                for value in feature_values:
                    left_mask = feature_values == value
                    right_mask = ~left_mask
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    info_gain = self._information_gain(y, left_y, right_y)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_feature = feature_index
                        best_threshold = value
                        best_left = left_y
                        best_right = right_y

        if max_info_gain == 0:
            return np.bincount(y).argmax()

        # Handle empty subsets by assigning the majority class of the parent
        if len(best_left) == 0:
            best_left = np.full_like(y[:1], np.bincount(y).argmax())
        if len(best_right) == 0:
            best_right = np.full_like(y[:1], np.bincount(y).argmax())

        # Recursively apply ID3 to the left and right subsets
        tree = {'feature': best_feature, 'threshold': best_threshold}
        if np.issubdtype(X[:, best_feature].dtype, np.number):  # Continuous data
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            tree['left'] = self._id3(X[left_mask], best_left)
            tree['right'] = self._id3(X[right_mask], best_right)
        else:  # Categorical data
            left_mask = X[:, best_feature] == best_threshold
            right_mask = ~left_mask
            tree['left'] = self._id3(X[left_mask], best_left)
            tree['right'] = self._id3(X[right_mask], best_right)

        return tree

    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-6))  # Adding small epsilon to avoid log(0)
        return entropy

    def _information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self._entropy(parent_y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)
        weighted_avg_entropy = left_weight * left_entropy + right_weight * right_entropy
        return parent_entropy - weighted_avg_entropy

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            feature_value = x[tree['feature']]
            if np.issubdtype(feature_value, np.number):  # Continuous data
                if feature_value <= tree['threshold']:
                    return self._predict_single(x, tree['left'])
                else:
                    return self._predict_single(x, tree['right'])
            else:  # Categorical data
                if feature_value == tree['threshold']:
                    return self._predict_single(x, tree['left'])
                else:
                    return self._predict_single(x, tree['right'])
        else:
            return tree


