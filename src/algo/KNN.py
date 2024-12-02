import numpy as np
from collections import Counter


# K-Nearest Neighbors (KNN)
class KNN:
    def __init__(self, k=3, metric='euclidean', p=3):
        self.k = k
        self.metric = metric
        self.p = p

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def minkowski_distance(self, x1, x2, p=3):
        return np.power(np.sum(np.abs(x1 - x2) ** p), 1 / p)

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self.minkowski_distance(x1, x2, self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = [self._distance(x_test, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

