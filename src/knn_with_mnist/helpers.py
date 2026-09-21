import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def show_training_image(X_train, y_train, index):
    image = X_train[index].reshape(28, 28)

    plt.imshow(image, cmap="gray")
    plt.title(f"🎯 Training example #{index} — Label: {y_train[index]}")
    plt.axis("off")
    plt.show()


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def plot_test_data(X_test, y_test):

    print("Plotting test data...")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    pca = PCA(n_components=2)

    X_test_2d = pca.fit_transform(X_test)

    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap="tab10")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Test Data")
    plt.colorbar(label="Digit")
    plt.show()


def number_of_dark_pixels(X):
    return np.sum(X < 128, axis=1)
