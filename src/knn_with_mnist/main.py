import numpy as np
import pandas as pd

from helpers import euclidean_distance, manhattan_distance
from knn import KNN


# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

y_train = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values

y_test = test.iloc[:, 0].values
X_test = test.iloc[:, 1:].values


# Values of k to test
k_values = [1, 3, 5, 10, 15]


# Test Euclidean and Manhattan
for distance_name, distance_function in [
    ("Euclidean", euclidean_distance),
    ("Manhattan", manhattan_distance),
]:
    print()
    print("================================")
    print(distance_name)
    print("================================")

    for k in k_values:
        # Create KNN with current k
        clf = KNN(k=k)

        # Train
        clf.fit(X_train, y_train)

        # Predict
        predictions = clf.predict(X_test, distance_metric=distance_function)

        # Accuracy
        correct = np.sum(np.array(predictions) == y_test)

        accuracy = correct / len(y_test)

        print("k =", k, "| Correct:", correct, "| Accuracy:", accuracy * 100, "%")
