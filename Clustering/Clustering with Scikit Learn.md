# Clustering with Scikit-Learn

Clustering is an unsupervised learning technique used to group similar data points into clusters. Scikit-learn provides various clustering algorithms and utilities to facilitate this process. This guide covers the basic steps to perform clustering using scikit-learn.

## 1. Introduction to Clustering Algorithms

Scikit-learn offers several clustering algorithms, including:

- **K-Means Clustering**
- **Hierarchical Clustering**
- **DBSCAN**

### 1.1 K-Means Clustering

K-Means is a widely used clustering algorithm that partitions the dataset into \( k \) clusters. Each cluster is defined by its centroid, which is the mean of the data points in the cluster.

### 1.2 Hierarchical Clustering

Hierarchical clustering creates a tree-like structure of clusters. It can be used to form nested clusters by merging or splitting clusters iteratively.

### 1.3 DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together closely packed data points and identifies points that are considered noise.

## 2. K-Means Clustering

### 2.1 Installation

To use K-Means clustering, you need to install scikit-learn. If not already installed, you can install it using pip:

```bash
pip install scikit-learn
```

### 2.2 Basic Usage

Here's a basic example of how to perform K-Means clustering using scikit-learn:

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample dataset
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Initialize K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model
kmeans.fit(X)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Get cluster labels
labels = kmeans.labels_

print("Cluster centroids:", centroids)
print("Cluster labels:", labels)
```

### 2.3 Parameters

- `n_clusters`: Number of clusters to form.
- `init`: Method for initialization. Options include `'k-means++'` (default) and `'random'`.
- `max_iter`: Maximum number of iterations for the algorithm to converge.
- `random_state`: Seed for random number generation.

### 2.4 Visualization

You can visualize clusters using libraries like `matplotlib`:

```python
import matplotlib.pyplot as plt

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
```

## 3. Hierarchical Clustering

### 3.1 Installation

Hierarchical clustering can be performed using the `scipy` library:

```bash
pip install scipy
```

### 3.2 Basic Usage

Here's an example of hierarchical clustering using `scipy`:

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Perform hierarchical clustering
Z = linkage(X, 'ward')

# Plot dendrogram
plt.figure()
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()

# Form clusters
labels = fcluster(Z, t=2, criterion='maxclust')
print("Cluster labels:", labels)
```

### 3.3 Parameters

- `method`: The linkage criterion to use. Options include `'single'`, `'complete'`, `'average'`, and `'ward'`.
- `t`: The number of clusters to form (for `fcluster`).

## 4. DBSCAN

### 4.1 Installation

To use DBSCAN, make sure you have scikit-learn installed:

```bash
pip install scikit-learn
```

### 4.2 Basic Usage

Here's an example of DBSCAN clustering:

```python
from sklearn.cluster import DBSCAN
import numpy as np

# Sample dataset
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Initialize DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2)

# Fit the model
labels = dbscan.fit_predict(X)

print("Cluster labels:", labels)
```

### 4.3 Parameters

- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.

## 5. Evaluation Metrics

### 5.1 Silhouette Score

The silhouette score measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where a higher value indicates better-defined clusters.

```python
from sklearn.metrics import silhouette_score

# Compute silhouette score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)
```

### 5.2 Adjusted Rand Index

The adjusted Rand index (ARI) measures the similarity between two data clusterings, considering the chance of random clustering.

```python
from sklearn.metrics import adjusted_rand_score

# True labels (for evaluation purposes)
true_labels = np.array([0, 0, 0, 1, 1, 1])

# Compute ARI
ari = adjusted_rand_score(true_labels, labels)
print("Adjusted Rand Index:", ari)
```

## 6. Conclusion

Scikit-learn provides powerful tools for clustering data, including K-Means, Hierarchical, and DBSCAN algorithms. By understanding the parameters and evaluation metrics, you can effectively group similar data points and gain insights from your data.

