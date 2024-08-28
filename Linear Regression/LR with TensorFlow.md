# Linear Regression with TensorFlow

**Linear Regression** is a basic machine learning technique used to model the relationship between a target variable and one or more input features. TensorFlow, a popular deep learning library, can also be used to implement linear regression models.

## Overview

Linear Regression in TensorFlow can be implemented using the `tf.keras` API, which provides high-level building blocks for creating machine learning models.

## Importing Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

## Data Preparation

1. **Load Data**

   ```python
   data = pd.read_csv('data.csv')
   ```

2. **Feature Selection and Target Variable**

   ```python
   X = data[['feature1', 'feature2']].values  # Input features
   y = data['target'].values  # Target variable
   ```

3. **Split Data**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

## Building the Model

1. **Initialize the Model**

   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
   ])
   ```

2. **Compile the Model**

   ```python
   model.compile(optimizer='adam', loss='mean_squared_error')
   ```

## Training the Model

```python
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

## Making Predictions

```python
y_pred = model.predict(X_test)
```

## Evaluating the Model

1. **Mean Squared Error (MSE)**

   ```python
   mse = mean_squared_error(y_test, y_pred)
   print(f'Mean Squared Error: {mse}')
   ```

2. **R-squared Score**

   ```python
   r2 = r2_score(y_test, y_pred)
   print(f'R-squared Score: {r2}')
   ```

## Model Coefficients

To retrieve the weights (coefficients) and bias (intercept) of the model:

```python
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(f'Weights: {weights}')
print(f'Bias: {bias}')
```

## Example Code

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('data.csv')

# Feature selection
X = data[['feature1', 'feature2']].values
y = data['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Model coefficients
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(f'Weights: {weights}')
print(f'Bias: {bias}')
```

## Summary

- **Linear Regression** models the relationship between input features and a target variable using a linear equation.
- TensorFlow provides a flexible `tf.keras` API for creating and training linear regression models.
- Evaluation metrics such as Mean Squared Error (MSE) and R-squared Score help assess the performance of the model.

## References

- [TensorFlow Linear Regression Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
- [Keras Dense Layer Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
