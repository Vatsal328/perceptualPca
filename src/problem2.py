# -*- coding: utf-8 -*-
"""B22CS059_problem2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GYDCo7DXDqJHsAZRucTrH6AvuHZlOEn8

# Problem 2
"""

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

"""## Task 1

### Loading the dataset
"""

# Fetch the dataset
data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

samples, h, w = data.images.shape
np.random.seed(42)

x = data.data
features = x.shape[1]

y = data.target
target_names = data.target_names
classes = target_names.shape[0]

print("Total dataset size:")
print("Samples: %d" % samples)
print("Features: %d" % features)
print( "Classes: %d" % classes)

"""### Splitting the dataset"""

# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""## Task 2

### Deciding value of n_comp
"""

pca = PCA().fit(X_train)

# Plot explained variance ratio to decide the value of n_components
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

"""### Implementing PCA"""

# Implement the PCA
pca = PCA(n_components = 140, whiten = True)
pca = pca.fit(X_train)

# Transform the training data using PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

"""## Task 3"""

# Using KNN classifier to classify
classifier = KNeighborsClassifier(n_neighbors=6) # (K = 6)

# Train the classifier using the transformed training data
clf = classifier.fit(X_train_pca, Y_train)

"""## Task 4

### Predicting the labels for test data
"""

# Use the trained classifier to make predictions on the transformed testing data
y_pred = classifier.predict(X_test_pca)

"""### Accuracy"""

# Report observations on model performance
print(classification_report(Y_test, y_pred, target_names=data.target_names))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test_pca, Y_test, display_labels=target_names, xticks_rotation="vertical")
plt.tight_layout()
plt.show()

"""### Visualising eigenfaces"""

# Visualize a subset of Eigenfaces
n_faces = 10
plt.figure(figsize=(10, 5))
for i in range(n_faces):
    plt.subplot(2, 5, i + 1)
    plt.imshow(pca.components_[i].reshape(data.images[0].shape), cmap='gray')
    plt.title(f"Eigenface {i+1}")
    plt.axis('off')
plt.show()