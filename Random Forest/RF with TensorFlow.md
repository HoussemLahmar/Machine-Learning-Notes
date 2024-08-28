# Random Forest with TensorFlow Decision Forests (TF-DF)

## Overview

TensorFlow Decision Forests (TF-DF) is an extension for TensorFlow that provides implementations of decision forest algorithms, including Random Forests. It is designed to work seamlessly with TensorFlow and is highly optimized for performance.

## 1. Introduction to TF-DF

### Definition

TensorFlow Decision Forests is a library for decision forest algorithms such as Random Forests and Gradient Boosted Trees. It allows you to train and evaluate these models using TensorFlow's infrastructure.

### Key Concepts

- **Decision Forests**: Ensemble models that consist of multiple decision trees.
- **Random Forest**: An ensemble method that combines multiple decision trees to improve predictive performance.

## 2. Installation

To use TF-DF, you need to install the library. You can install it using pip:

```bash
pip install tensorflow_decision_forests
```

## 3. Basic Usage

### Importing Libraries

```python
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### Loading Data

For this example, we will use the Iris dataset:

```python
from sklearn.datasets import load_iris

# Load the dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

### Splitting Data

```python
# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Convert DataFrames to TensorFlow datasets
train_data = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, task=tfdf.keras.ClassificationTask(label="target"))
test_data = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.ClassificationTask(label="target"))
```

### Training the Model

```python
# Initialize the Random Forest model
model = tfdf.keras.RandomForestModel(task=tfdf.keras.ClassificationTask(label="target"), num_trees=100)

# Train the model
model.fit(train_data)
```

### Making Predictions

```python
# Make predictions
predictions = model.predict(test_data)

# Convert predictions to a DataFrame
predicted_labels = pd.Series([x[0] for x in predictions], name="Predicted")

# Combine with actual labels
results_df = test_df.reset_index(drop=True)
results_df["Predicted"] = predicted_labels
```

### Evaluating the Model

```python
# Calculate accuracy
accuracy = accuracy_score(test_df["target"], results_df["Predicted"])
print(f'Accuracy: {accuracy:.2f}')
```

## 4. Hyperparameter Tuning

TF-DF supports various hyperparameters for tuning, such as:

- **`num_trees`**: Number of trees in the forest.
- **`max_depth`**: Maximum depth of the trees.
- **`min_examples`**: Minimum number of examples required to split a node.
- **`max_features`**: Number of features to consider when looking for the best split.

#### Example of Hyperparameter Tuning

```python
# Initialize the Random Forest model with hyperparameters
model = tfdf.keras.RandomForestModel(
    task=tfdf.keras.ClassificationTask(label="target"),
    num_trees=100,
    max_depth=10,
    min_examples=5
)

# Train the model
model.fit(train_data)
```

## 5. Working with Datasets

TF-DF can be applied to various datasets, including:

- **Classification Datasets**: Iris, Wine, Breast Cancer, etc.
- **Regression Datasets**: Boston Housing, Energy Efficiency, etc.

## 6. Summary

TensorFlow Decision Forests provides a robust implementation of Random Forests within the TensorFlow ecosystem. By utilizing TF-DF, you can leverage TensorFlow's performance optimizations and seamlessly integrate decision forests into your machine learning workflows. Understanding how to implement, tune, and evaluate Random Forest models with TF-DF can greatly enhance your machine learning projects.
