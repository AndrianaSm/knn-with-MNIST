# 🤖 Exercise 1 – MNIST Classification with KNN

## 🎯 Goal

The goal was to implement a KNN classifier and use it to classify handwritten digits from MNIST.

Each image is **28 × 28 pixels = 784 features**. I used the pixel values directly as features.

I tested:

- 📏 Euclidean distance
- 📐 Manhattan distance
- 🔢 `k = {1, 3, 5, 10, 15}`

## 📊 Results

<table>
<tr>
<td>

### 📏 Euclidean

| k | Correct | Accuracy |
|---:|---:|---:|
| 1 | 885 | 88.59% |
| 3 | **889** | **88.99%** |
| 5 | 883 | 88.39% |
| 10 | 872 | 87.29% |
| 15 | 846 | 84.68% |

</td>
<td>

### 📐 Manhattan

| k | Correct | Accuracy |
|---:|---:|---:|
| 1 | 874 | 87.49% |
| 3 | 864 | 86.49% |
| 5 | 867 | 86.79% |
| 10 | 845 | 84.58% |
| 15 | 814 | 81.48% |

</td>
</tr>
</table>

## 🔍 Observations

Euclidean distance had higher accuracy than Manhattan distance for all tested values of `k`.

For Euclidean distance, accuracy was highest at `k = 3` with **88.99%**. Accuracy generally decreased for larger `k` values.

For Manhattan distance, the highest accuracy was at `k = 1` with **87.49%**.

A possible explanation is that larger `k` values include more distant and less similar training examples in the majority vote.

## 🗺️ PCA Visualization

I also used PCA to reduce the 784-dimensional test data to 2 dimensions for visualization.

Each dot represents one test image, and the color represents its digit.

<img width="1512" height="913" alt="Figure_1" src="https://github.com/user-attachments/assets/c6e8b3f2-aa65-45e7-b4aa-64dc98b74696" />


The plot shows that some digits form relatively clear groups, while others overlap. Since PCA reduces 784 dimensions to only 2, some information is lost, so this plot is mainly useful for getting a visual idea of the dataset.

## 🧩 Confusion Matrix

I generated a confusion matrix for **Euclidean distance with `k = 3`**.

- ↔️ Rows = actual digit
- ⬇️ Columns = predicted digit
- 🟩 Diagonal = correct predictions
- ❌ Off-diagonal values = mistakes

<img width="1512" height="819" alt="ma" src="https://github.com/user-attachments/assets/0c852592-5519-4c29-b9a3-8dff7e90ff3d" />

### ❌ Most common mistakes

The biggest confusions were:

| Actual | Predicted | Mistakes |
|---:|---:|---:|
| 4 | 9 | **16** |
| 7 | 9 | **7** |
| 2 | 1 | **6** |
| 3 | 5 | **6** |
| 2 | 7 | **5** |
| 5 | 3 | **5** |
| 8 | 1 | **5** |
| 8 | 5 | **5** |

The most noticeable error was **4 → 9**, where 16 images of the digit 4 were classified as 9. This shows that some handwritten digits can look quite similar to the classifier.

### 👀 Visualizing wrong predictions

To better understand these errors, I also visualized some of the misclassified images.

For example, the following visualization shows images that were actually **7** but were predicted as **9**:


<img width="1200" height="300" alt="mistake7" src="https://github.com/user-attachments/assets/b2a67694-4579-4a5b-9b39-69a2bbf21a1b" />


This makes it easier to see why some handwritten digits can be difficult to distinguish.


## 🧠 Conclusion

The best result in this experiment was **88.99% accuracy using Euclidean distance with `k = 3`**.

The experiment also showed that increasing `k` did not necessarily improve the results. Overall, the exercise helped demonstrate how KNN uses distances and majority voting to classify images.
