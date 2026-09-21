import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helpers import euclidean_distance, manhattan_distance
from knn import KNN
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

# ============================================================
# 1. Load data
# ============================================================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

y_train = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values

y_test = test.iloc[:, 0].values
X_test = test.iloc[:, 1:].values


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 2. Values of k to test
# ============================================================

k_values = [1, 3, 5, 10, 15]


# ============================================================
# 3. Run experiments
# ============================================================

results = []

for distance_name, distance_function in [
    ("Euclidean", euclidean_distance),
    ("Manhattan", manhattan_distance),
]:
    print()
    print("========================================")
    print(distance_name)
    print("========================================")

    for k in k_values:
        print()
        print("Testing k =", k)

        # Create KNN
        clf = KNN(k=k)

        # Train
        clf.fit(X_train, y_train)

        # Predict
        predictions = clf.predict(
            X_test,
            distance_metric=distance_function,
        )

        # Accuracy
        correct = np.sum(np.array(predictions) == y_test)

        accuracy = correct / len(y_test)

        print("Correct predictions:", correct)
        print("Accuracy:", accuracy * 100, "%")

        results.append(
            (
                distance_name,
                k,
                correct,
                accuracy * 100,
            )
        )


# ============================================================
# 4. Print summary
# ============================================================

print()
print("========================================")
print("SUMMARY")
print("========================================")

for distance, k, correct, accuracy in results:
    print(
        f"{distance:10} | "
        f"k = {k:2} | "
        f"Correct = {correct:3} | "
        f"Accuracy = {accuracy:.2f}%"
    )


# ============================================================
# 5. Confusion matrix
# ============================================================

# Example: Euclidean distance with k = 3

clf = KNN(k=3)

clf.fit(X_train, y_train)

euclidean_predictions = clf.predict(
    X_test,
    distance_metric=euclidean_distance,
)

cm = confusion_matrix(
    y_test,
    euclidean_predictions,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10),
)

disp.plot()

plt.title("Confusion Matrix - Euclidean, k=3")
plt.show()

print("Test samples:", len(y_test))
print("Confusion matrix total:", cm.sum())
