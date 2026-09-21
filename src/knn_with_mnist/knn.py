from collections import Counter
import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, distance_metric):
        predictions = [self._predict(x, distance_metric) for x in X]

        return predictions

    def _predict(self, x, distance_metric):

        distances = [distance_metric(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        prediction = Counter(k_nearest_labels).most_common(1)[0][0]

        return prediction
