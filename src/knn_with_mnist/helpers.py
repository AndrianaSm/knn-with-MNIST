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


def show_misclassified_images(
    X_test, y_test, predictions, actual_label, predicted_label, n=8
):
    wrong_indices = np.where(
        (y_test == actual_label) & (np.array(predictions) == predicted_label)
    )[0]

    n = min(n, len(wrong_indices))

    plt.figure(figsize=(12, 3))

    for i in range(n):
        index = wrong_indices[i]

        plt.subplot(1, n, i + 1)
        plt.imshow(X_test[index].reshape(28, 28), cmap="gray")

        plt.title(f"Act: {y_test[index]}\nPred: {predictions[index]}")

        plt.axis("off")

    plt.suptitle(f"❌ Misclassified {actual_label} → {predicted_label}")

    plt.tight_layout()
    plt.show()
