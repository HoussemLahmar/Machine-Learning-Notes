# SVM with scikit-learn

### 1. Installing scikit-learn

Make sure you have scikit-learn installed. You can install it using pip:

```bash
pip install scikit-learn
```

### 2. Importing Required Libraries

Start by importing the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

### 3. Loading the Data

You can use the `iris` dataset from scikit-learn for demonstration:

```python
# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

### 4. Splitting the Data

Split the dataset into training and testing sets:

```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 5. Linear SVM Classification

#### a. Creating and Training the Model

```python
# Create a linear SVM classifier
linear_svm = SVC(kernel='linear', C=1)

# Train the classifier
linear_svm.fit(X_train, y_train)
```

#### b. Making Predictions

```python
# Make predictions
y_pred = linear_svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Linear SVM Accuracy: {accuracy:.2f}")
```

### 6. Nonlinear SVM Classification

#### a. Using Polynomial Kernel

```python
# Create a polynomial SVM classifier
poly_svm = SVC(kernel='poly', degree=3, C=1)

# Train the classifier
poly_svm.fit(X_train, y_train)

# Make predictions
y_pred_poly = poly_svm.predict(X_test)

# Evaluate accuracy
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Polynomial Kernel SVM Accuracy: {accuracy_poly:.2f}")
```

#### b. Using Radial Basis Function (RBF) Kernel

```python
# Create an RBF SVM classifier
rbf_svm = SVC(kernel='rbf', gamma=0.5, C=1)

# Train the classifier
rbf_svm.fit(X_train, y_train)

# Make predictions
y_pred_rbf = rbf_svm.predict(X_test)

# Evaluate accuracy
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF Kernel SVM Accuracy: {accuracy_rbf:.2f}")
```

### 7. Hyperparameter Tuning

Use Grid Search to find the best parameters:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.5, 1],
    'kernel': ['rbf']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters found:", grid_search.best_params_)

# Evaluate on test set
y_pred_grid = grid_search.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)
print(f"Grid Search SVM Accuracy: {accuracy_grid:.2f}")
```

### 8. Visualizing Decision Boundaries (For 2D Data)

If your dataset is 2D, you can visualize the decision boundaries:

```python
def plot_decision_boundaries(X, y, model, title):
    plt.figure(figsize=(8, 6))
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# For visualization, use a 2D dataset or reduce dimensions
# Example usage:
# plot_decision_boundaries(X_train[:, :2], y_train, "SVM Decision Boundary")
```

### Summary

1. **Install scikit-learn** and import necessary libraries.
2. **Load and split data** into training and testing sets.
3. **Train and evaluate** linear and nonlinear SVMs using different kernels.
4. **Tune hyperparameters** using Grid Search.
5. **Visualize decision boundaries** if working with 2D data.

