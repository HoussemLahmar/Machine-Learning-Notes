 # Clustering Course Notes

## 1. How Clustering Works

Clustering is a type of unsupervised machine learning algorithm that groups similar data points into clusters. The goal of clustering is to divide the dataset into a number of groups so that data points within each group are more comparable to one another and different from those in other groups.

### Mathematical Algorithm

Given a dataset \( X \) with \( n \) data points and \( m \) features, the clustering algorithm aims to partition the dataset into \( k \) clusters, where \( k \) is a hyperparameter.

The clustering process can be represented mathematically as:

$$
X = \{x_1, x_2, \ldots, x_n\}
$$

$$
C = \{c_1, c_2, \ldots, c_k\}
$$

where \( C \) represents the set of cluster centroids.

### Logical Algorithm

1. **Initialize Clusters:** Initialize \( k \) clusters with random centroids.
2. **Assign Data Points:** Assign each data point to the closest cluster centroid based on the Euclidean distance.
3. **Update Centroids:** Update the cluster centroids as the mean of all data points assigned to each cluster.
4. **Repeat:** Repeat steps 2-3 until convergence or a stopping criterion is reached.

## 2. Euclidean Distance

Euclidean distance is a measure of the straight-line distance between two points in \( n \)-dimensional space. It is commonly used in clustering algorithms to calculate the distance between data points and cluster centroids.

### Mathematical Algorithm

The Euclidean distance between two points \( x \) and \( y \) can be calculated as:

$$
d(x, y) = \sqrt{\sum (x - y)^2}
$$

where \( d(x, y) \) represents the Euclidean distance between \( x \) and \( y \).

### Logical Algorithm

1. **Calculate Differences:** Calculate the differences between the corresponding features of \( x \) and \( y \).
2. **Square Differences:** Square each difference.
3. **Sum Squared Differences:** Sum the squared differences.
4. **Take Square Root:** Take the square root of the sum.

## 3. K-Means Clustering

K-means clustering is a popular algorithm that partitions the dataset into \( k \) clusters based on the Euclidean distance between data points and cluster centroids.

### Mathematical Algorithm

1. Initialize \( k \) cluster centroids randomly.
2. Assign each data point to the closest cluster centroid based on the Euclidean distance.
3. Update the cluster centroids as the mean of all data points assigned to each cluster.
4. Repeat steps 2-3 until convergence or a stopping criterion is reached.

### Logical Algorithm

1. **Initialize Centroids:** Initialize \( k \) cluster centroids with random values.
2. **Assign Data Points:** Assign each data point to the closest cluster centroid based on the Euclidean distance.
3. **Update Centroids:** Update the cluster centroids as the mean of all data points assigned to each cluster.
4. **Convergence Check:** Check for convergence using a stopping criterion, such as the change in cluster centroids or the number of iterations.

## 4. Feature Normalization

Feature normalization is a preprocessing step that scales the features of the dataset to a common range, usually between 0 and 1. This is essential to prevent features with large ranges from dominating the clustering process.

### Mathematical Algorithm

The feature normalization process can be represented mathematically as:

$$
x' = \frac{x - \text{min}}{\text{max} - \text{min}}
$$

where \( x' \) represents the normalized feature value, \( x \) represents the original feature value, \( \text{min} \) represents the minimum value of the feature, and \( \text{max} \) represents the maximum value of the feature.

### Logical Algorithm

1. **Calculate Min and Max:** Calculate the minimum and maximum values of each feature.
2. **Subtract Min:** Subtract the minimum value from each feature value.
3. **Divide by Range:** Divide each feature value by the range of the feature (\(\text{max} - \text{min}\)).
4. **Normalize:** Normalize each feature value to the range [0, 1].

## 5. Working with Datasets

Working with datasets involves loading, exploring, and preprocessing the data before applying clustering algorithms.

### Logical Algorithm

1. **Load Dataset:** Load the dataset into a suitable data structure, such as a pandas DataFrame.
2. **Explore Dataset:** Explore the dataset to understand the distribution of features and identify any missing values or outliers.
3. **Preprocess Dataset:** Preprocess the dataset by handling missing values, encoding categorical variables, and scaling the features.

## 6. Cluster Interpretation

Cluster interpretation involves analyzing the resulting clusters to identify patterns, trends, and insights.

### Logical Algorithm

1. **Analyze Cluster Centroids:** Analyze the cluster centroids to understand the characteristics of each cluster.
2. **Visualize Clusters:** Visualize the clusters using dimensionality reduction techniques, such as PCA or t-SNE, to identify patterns and relationships.
3. **Identify Outliers:** Identify outliers and anomalies within each cluster.
4. **Draw Conclusions:** Draw conclusions about the clusters and identify potential applications or insights.

## 7. Summary

Clustering is a powerful unsupervised machine learning algorithm that groups similar data points into clusters. Key concepts include Euclidean distance, K-means clustering, feature normalization, working with datasets, and cluster interpretation. By applying these concepts, we can uncover hidden patterns and insights in datasets and make informed decisions.

### Mathematical Algorithm Recap

The clustering process can be represented mathematically as:

$$
X = \{x_1, x_2, \ldots, x_n\}
$$

$$
C = \{c_1, c_2, \ldots, c_k\}
$$

where \( C \) represents the set of cluster centroids.

### Logical Algorithm Recap

1. **Initialize Clusters:** Initialize \( k \) clusters with random centroids.
2. **Assign Data Points:** Assign each data point to the closest cluster centroid based on the Euclidean distance.
3. **Update Centroids:** Update the cluster centroids as the mean of all data points assigned to each cluster.
4. **Repeat:** Repeat steps 2-3 until convergence or a stopping criterion is reached.

## 8. Advanced Topics in Clustering

### 8.1 Hierarchical Clustering

Hierarchical clustering builds a hierarchy of clusters by merging or splitting existing clusters.

#### Mathematical Algorithm

The hierarchical clustering algorithm can be represented mathematically as:

$$
H = \{h_1, h_2, \ldots, h_k\}
$$

where \( H \) represents the hierarchy of clusters.

#### Logical Algorithm

1. **Initialize Clusters:** Initialize each data point as a separate cluster.
2. **Calculate Distances:** Calculate the distances between each pair of clusters.
3. **Merge Clusters:** Merge the two closest clusters based on the distance metric.
4. **Repeat:** Repeat steps 2-3 until a stopping criterion is reached.

### 8.2 DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups data points into clusters based on density and proximity.

#### Mathematical Algorithm

The DBSCAN algorithm can be represented mathematically as:

- $\( \text{Eps} \)$ = maximum distance between points in a cluster
- $\( \text{MinPts} \)$ = minimum number of points required to form a dense region

#### Logical Algorithm

1. **Initialize Clusters:** Initialize each data point as a separate cluster.
2. **Find Neighbors:** Find all neighbors within a distance of Eps for each data point.
3. **Form Clusters:** Form clusters by grouping data points with at least MinPts neighbors.
4. **Identify Noise:** Identify noise points that do not belong to any cluster.

## 9. Clustering Evaluation Metrics

Clustering evaluation metrics measure the quality of clustering results.

### 9.1 Silhouette Coefficient

The Silhouette Coefficient measures how similar an object is to its own cluster compared to other clusters.

#### Mathematical Algorithm

The Silhouette Coefficient can be calculated as:

$$
s = \frac{b - a}{\max(a, b)}
$$

where \( a \) is the average distance to the points in the same cluster, and \( b \) is the average distance to the points in the nearest cluster.

### 9.2 Calinski-Harabasz Index

The Calinski-Harabasz Index measures the ratio of between-cluster variance to within-cluster variance.

#### Mathematical Algorithm

The Calinski-Harabasz Index can be calculated as:

$$
\text{CH} = \frac{B / W \cdot (N - k)}{k - 1}
$$

where \( B \) is the between-cluster variance, \( W \) is the within-cluster variance, \( N \) is the total number of data points, and \( k \) is the number of clusters.

## 10. Conclusion

Clustering is a powerful unsupervised machine learning technique that groups similar data points into clusters. Understanding the mathematical and logical algorithms behind clustering helps in applying these techniques to real-world problems.

### Python Implementation

Here is a Python implementation of the K-means clustering algorithm using scikit-learn:

```python
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Initialize K-means with k=2
kmeans = KMeans(n_clusters=2)

# Fit the model
kmeans.fit(X)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Get cluster labels
labels = kmeans.labels_

print("Cluster Centroids:")
print(centroids)
print("Cluster Labels:")
print(labels)
```
