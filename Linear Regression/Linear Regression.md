### Linear Regression


# Simple Linear Regression

**Simple Linear Regression** is a type of supervised learning algorithm that predicts a continuous output variable based on a single input feature. The goal is to create a linear equation that best predicts the value of the target variable.

## Equation

The simple linear regression equation is:

```math
y = β0 + β1x + ε
```

where:

- `y` is the target variable (output)
- `x` is the input feature (predictor)
- `β0` is the intercept or bias term
- `β1` is the slope coefficient
- `ε` is the error term (residual)

## Vectorization

To perform linear regression on a dataset, we can represent the data using vectors and matrices. Let's denote the input feature vector as `X` and the target variable vector as `y`. We can then write the linear regression equation in matrix form:

```
y = Xβ + ε
```

where:

- `X` is an `n x 2` matrix, where `n` is the number of samples, and each row represents a data point with a bias term (1) and the input feature value
- `β` is a `2 x 1` vector, containing the intercept and slope coefficients
- `ε` is an `n x 1` vector, representing the error terms

## Important Functions and Equations

### Cost Function (Mean Squared Error - MSE)

The cost function measures the average squared difference between predicted and actual values. The goal is to minimize the cost function.

```
J(β) = (1/2n) * ∑(y_i - (β0 + β1x_i))^2
```

where `J(β)` is the cost function, `n` is the number of samples, and `y_i` and `x_i` are the i-th target variable and input feature values, respectively.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively using the following equations:

```
β0_new = β0_old - α * (1/n) * ∑(-2 * (y_i - (β0_old + β1_old * x_i)))
β1_new = β1_old - α * (1/n) * ∑(-2 * x_i * (y_i - (β0_old + β1_old * x_i)))
```

where `α` is the learning rate, and `β0_old` and `β1_old` are the previous values of the intercept and slope coefficients, respectively.

### Evaluating the Fitness of the Model with a Cost Function

To evaluate the fitness of the model, we can use the cost function (MSE) to measure the average squared difference between predicted and actual values. A lower cost function value indicates a better fit.

### Solving OLS for Simple Linear Regression

Ordinary Least Squares (OLS) is a method for estimating the model parameters that minimizes the sum of the squared errors. The OLS solution for simple linear regression is:

```
β1 = ∑((x_i - x_mean) * (y_i - y_mean)) / ∑((x_i - x_mean)^2)
β0 = y_mean - β1 * x_mean
```

where `x_mean` and `y_mean` are the means of the input feature and target variable, respectively.

### Evaluating the Model

To evaluate the model, we can use metrics such as:

- **Coefficient of Determination (R-squared):**
  Measures the proportion of the variance in the target variable that is predictable from the input feature.

  ```
  R^2 = 1 - (SSE / SST)
  ```

  where `SSE` is the sum of the squared errors, and `SST` is the total sum of squares.

- **Mean Absolute Error (MAE):**
  Measures the average absolute difference between predicted and actual values.

  ```
  MAE = (1/n) * ∑|y_i - (β0 + β1x_i)|
  ```

## Multiple Linear Regression

**Multiple Linear Regression** is an extension of simple linear regression, where multiple input features are used to predict the target variable.

### Equation

The multiple linear regression equation is:

```
y = β0 + β1x1 + β2x2 + … + βnxn + ε
```

where `x1`, `x2`, …, `xn` are the input features, and `β1`, `β2`, …, `βn` are the corresponding coefficients.

### Vectorization

The multiple linear regression equation can be written in matrix form:

```
y = Xβ + ε
```

where:

- `X` is an `n x (p+1)` matrix, where `n` is the number of samples, `p` is the number of input features, and each row represents a data point with a bias term (1) and the input feature values
- `β` is a `(p+1) x 1` vector, containing the intercept and slope coefficients
- `ε` is an `n x 1` vector, representing the error terms

### Important Functions and Equations

#### Cost Function (Mean Squared Error - MSE)

The cost function measures the average squared difference between predicted and actual values. The goal is to minimize the cost function.

```
J(β) = (1/2n) * ∑(y_i - (β0 + β1x1_i + β2x2_i + … + βnxn_i))^2
```

where `J(β)` is the cost function, `n` is the number of samples, and `y_i`, `x1_i`, `x2_i`, …, `xn_i` are the i-th target variable and input feature values, respectively.

#### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively using the following equations:

```
β0_new = β0_old - α * (1/n) * ∑(-2 * (y_i - (β0_old + β1_old * x1_i + β2_old * x2_i + … + βn_old * xn_i)))
β1_new = β1_old - α * (1/n) * ∑(-2 * x1_i * (y_i - (β0_old + β1_old * x1_i + β2_old * x2_i + … + βn_old * xn_i)))
β2_new = β2_old - α * (1/n) * ∑(-2 * x2_i * (y_i - (β0_old + β1_old * x1_i + β2_old * x2_i + … + βn_old * xn_i)))
...
βn_new = βn_old - α * (1/n) * ∑(-2 * xn_i * (y_i - (β0_old + β1_old * x1_i + β2_old * x2_i + … + βn_old * xn_i)))
```

where `α` is the learning rate, and `β0_old`, `β1_old`, `β2_old`, …, `βn_old` are the previous values of the intercept and slope coefficients, respectively.

## Polynomial Regression

**Polynomial Regression** is a type of regression analysis where the relationship between the independent variable and the dependent variable is modeled as an nth degree polynomial.

### Equation

The polynomial regression equation is:

```
y = β0 + β1x + β2x^2 + … + βnx^n + ε
```

where `x` is the input feature, and `β1`, `β2`, …, `βn` are the corresponding coefficients.

### Vectorization

The polynomial regression equation can be written in matrix form:

```
y = Xβ + ε
```

where:

- `X` is an `n x (n+1)` matrix, where `n` is the number of samples, and each row represents a data point with a bias term (1) and the input feature values raised to the powers of 1, 2, …, n
- `β` is an `(n+1) x 1` vector, containing the intercept and slope coefficients
- `ε` is an `n x 1` vector, representing the error terms

## Applying Linear Regression

Linear regression can be applied to a wide range of problems, including:

- **Predicting continuous outcomes:**
  Linear regression can be used to predict continuous outcomes, such as stock prices, temperatures, or blood pressure.

- **Analyzing relationships:**
  Linear regression can be used to analyze the relationship between two or more variables, such as the relationship between a dependent variable and one or more independent variables.

- **Identifying patterns:**
  Linear regression can be used to identify patterns in data, such as trends or correlations.

## Exploring the Data

Before applying linear regression, it's essential to explore the data to understand the relationships between the variables and to identify any patterns or trends.

### Important Functions and Equations

- **Correlation Coefficient:**
  The correlation coefficient measures the strength and direction of the linear relationship between two variables.

  ```
  ρ(X, Y) = cov(X, Y) / (σ_X * σ_Y)
  ```

  where `cov(X, Y)` is the covariance between `X` and `Y`, and `σ_X` and `σ_Y` are the standard deviations of `X` and `Y`, respectively.

- **Scatter Plots:**
  Scatter plots are used to visualize the relationship between two variables.



  ```
  Scatter Plot: y vs x
  ```

  where `y` is plotted on the y-axis and `x` is plotted on the x-axis.

## Conclusion

Linear regression is a powerful tool for predicting continuous outcomes and analyzing relationships between variables. By understanding the underlying equations, cost functions, and optimization techniques, you can effectively apply linear regression to a wide range of problems. Additionally, exploring the data and using visualization techniques can help you gain insights and improve the performance of your linear regression models.
