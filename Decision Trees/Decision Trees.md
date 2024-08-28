# Decision Tree 

## 1. Decision Tree

### Definition

A Decision Tree is a type of supervised learning algorithm used for both classification and regression tasks. It is a tree-like model that splits the data into subsets based on the values of input features.

### How it Works

A Decision Tree consists of:

- **Root Node**: The topmost node representing the entire dataset.
- **Decision Nodes**: Nodes that split the data into subsets based on a specific feature and value.
- **Leaf Nodes**: Nodes that represent the predicted class or value.

The algorithm works by recursively partitioning the data into smaller subsets until a stopping criterion is met.

### Decision Tree Schematic

```
          [Root Node]
           /      \
     [Decision]  [Decision]
       /    \       /    \
    [Leaf] [Leaf] [Leaf] [Leaf]
```

## 2. Nonlinear Classification and Regression

### Introduction

Decision Trees can handle both linear and nonlinear relationships between features and target variables. They are particularly useful when the relationship is complex and nonlinear.

### Advantages

- Handle high-dimensional data
- Model nonlinear relationships
- Handle missing values
- Provide interpretability

## 3. Training Decision Trees

### Algorithm

The training process involves the following steps:

1. **Root Node**: Select the entire dataset as the root node.
2. **Splitting**: Select a feature and a value to split the data into two subsets.
3. **Recursion**: Recursively apply steps 1-2 to each subset until a stopping criterion is met.
4. **Leaf Node**: Assign a class label or value to each leaf node.

### Stopping Criteria

- **Maximum Depth**: Stop when the tree reaches a maximum depth.
- **Minimum Samples**: Stop when the number of samples in a node falls below a minimum threshold.
- **Minimum Impurity**: Stop when the impurity of a node falls below a minimum threshold.

## 4. Selecting the Questions

### Feature Selection

The algorithm selects the best feature and value to split the data at each node. This is done by evaluating the information gain or Gini impurity for each feature and value.

### Feature Importance

The importance of each feature can be calculated by summing the information gain or Gini impurity for each feature across all nodes.

## 5. Information Gain

### Definition

Information Gain measures the reduction in impurity or uncertainty after splitting the data.

### Formula

$$ IG(X, y) = H(y) - H(y|X) $$

where \( H(y) \) is the entropy of the target variable, and \( H(y|X) \) is the conditional entropy of the target variable given the feature \( X \).

## 6. Gini Impurity

### Definition

Gini Impurity measures the impurity or uncertainty of a node.

### Formula

$$ Gini = 1 - \sum_{i=1}^{C} p_i^2 $$

where \( p_i \) is the proportion of samples belonging to class \( i \) in the node.

## 7. Implementation with Scikit-learn

### Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
```

## 8. Implementation with TensorFlow

### Code

```python
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Encode target variable
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple Decision Tree model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=5)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
y_true = tf.argmax(y_test, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Accuracy: {accuracy:.2f}')
```

## 9. Working with Datasets

### Salary Prediction

- **Dataset**: Salary data with features such as age, experience, education, and job title.
- **Task**: Predict the salary based on the input features.
- **Features**: Age, experience, education, job title, etc.

## 10. Summary

### Overview

Decision Trees are a powerful algorithm for both classification and regression tasks. They can handle nonlinear relationships, high-dimensional data, and missing values. By understanding the concepts of information gain, Gini impurity, and feature importance, we can build accurate and interpretable models using Decision Trees.
