"""
Title: Iris Dataset Clustering with KMeans
Description: Clustering the Iris dataset using KMeans algorithm and comparing it with true labels.
Author: Abolfazl Karimi
GitHub: https://github.com/abolfazlkarimi83
License: MIT
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot: KMeans Clustering Results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 2], X[:, 3], c=labels, s=15, cmap='viridis')
plt.scatter(centers[:, 2], centers[:, 3], s=60, c='red', marker='x', label='Centers')
plt.xlabel('Petal Length (X3)')
plt.ylabel('Petal Width (X4)')
plt.title('K-Means Clustering')
plt.legend()

# Plot: Ground Truth Labels
plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=y, s=15, cmap='viridis')
plt.xlabel('Petal Length (X3)')
plt.ylabel('Petal Width (X4)')
plt.title('Actual Classes (Ground Truth)')

plt.tight_layout()
plt.savefig("iris_kmeans_plot.png")  # save image
plt.show()
