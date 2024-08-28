# Naive Bayes with TensorFlow

TensorFlow is more commonly used for deep learning models, but we can implement a basic Naive Bayes classifier to illustrate its workings. This example will use TensorFlow operations to calculate probabilities manually.

## 1. Importing Libraries

First, import TensorFlow and other necessary libraries:

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
```

## 2. Loading the Data

For this example, we’ll use the Iris dataset:

```python
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
```

## 3. Preprocessing the Data

Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4. Defining the Naive Bayes Model

We’ll implement the Naive Bayes algorithm using TensorFlow operations:

```python
class NaiveBayes(tf.Module):
    def __init__(self):
        self.class_priors = None
        self.feature_likelihoods = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # Calculate class priors
        class_counts = np.bincount(y)
        self.class_priors = class_counts / num_samples

        # Calculate feature likelihoods
        self.feature_likelihoods = {}
        for cls in range(num_classes):
            X_cls = X[y == cls]
            feature_likelihoods = (np.mean(X_cls, axis=0), np.std(X_cls, axis=0))
            self.feature_likelihoods[cls] = feature_likelihoods

    def predict(self, X):
        num_classes = len(self.class_priors)
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            log_probs = np.zeros(num_classes)
            for cls in range(num_classes):
                prior = np.log(self.class_priors[cls])
                mean, std = self.feature_likelihoods[cls]
                likelihood = -0.5 * np.sum(((X[i] - mean) / std) ** 2)
                log_probs[cls] = prior + likelihood
            predictions[i] = np.argmax(log_probs)

        return predictions
```

## 5. Training and Testing the Model

Initialize, train, and test the Naive Bayes model:

```python
# Initialize the model
nb = NaiveBayes()

# Train the model
nb.fit(X_train, y_train)

# Predict on the test data
y_pred = nb.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## 6. Summary

Implementing Naive Bayes from scratch in TensorFlow can be an educational exercise to understand how Naive Bayes works. However, for practical purposes, using libraries like Scikit-Learn is recommended as they provide optimized and tested implementations.

### Advantages
- Demonstrates basic Naive Bayes principles with TensorFlow
- Custom implementation provides learning insights

### Disadvantages
- Not as efficient or optimized as Scikit-Learn implementations
- TensorFlow is more suited for deep learning models


