import numpy as np

# Posterior: P(y|X)
# Prior: P(y)
# Likelihood: ∏ P(X|y)


# Continuous Data
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.variance[cls] = X_c.var(axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def gaussian_pdf(self, x, mean, var):
        eps = 1e-6
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                conditional = np.sum(np.log(self.gaussian_pdf(x, self.mean[cls], self.variance[cls])))
                posteriors[cls] = prior + conditional
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

# Discrete Data
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_counts = {}
        self.feature_counts = {}
        self.priors = {}

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.class_counts[cls] = X_c.shape[0]
            self.feature_counts[cls] = X_c.sum(axis=0)  # Sum of features in each class
            self.priors[cls] = self.class_counts[cls] / num_samples  # Class prior probability

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        posteriors = {}
        # Use log instead of multiplication to improve numerical stability for reducing underflow chances
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.sum(x * np.log(self.feature_counts[cls] + 1))  # Laplace smoothing
            posteriors[cls] = prior + likelihood
        
        return max(posteriors, key=posteriors.get)