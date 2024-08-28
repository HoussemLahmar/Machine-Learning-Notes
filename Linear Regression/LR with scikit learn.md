# Linear Regression with scikit-learn

**Linear Regression** is a fundamental machine learning technique used to model the relationship between a target variable and one or more input features. `scikit-learn` provides an efficient implementation for performing linear regression.

## Overview

Linear Regression models the relationship between a target variable \( y \) and one or more input features \( X \) by fitting a linear equation. In `scikit-learn`, this is achieved using the `LinearRegression` class from the `sklearn.linear_model` module.

## Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## Data Preparation

1. **Load Data**

   ```python
   data = pd.read_csv('data.csv')
   ```

2. **Feature Selection and Target Variable**

   ```python
   X = data[['feature1', 'feature2']]  # Input features
   y = data['target']  # Target variable
   ```

3. **Split Data**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

## Creating and Training the Model

1. **Initialize the Model**

   ```python
   model = LinearRegression()
   ```

2. **Fit the Model**

   ```python
   model.fit(X_train, y_train)
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

1. **Coefficients and Intercept**

   ```python
   print(f'Coefficients: {model.coef_}')
   print(f'Intercept: {model.intercept_}')
   ```

## Example Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('data.csv')

# Feature selection
X = data[['feature1', 'feature2']]
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
```

## Summary

- **Linear Regression** models the relationship between input features and a target variable using a linear equation.
- `scikit-learn` provides an easy-to-use `LinearRegression` class for creating and training linear models.
- Evaluation metrics such as Mean Squared Error (MSE) and R-squared Score help assess the performance of the model.

## References

- [scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
