
# Linear Regression

## Simple LR
**Simple Linear Regression** is a type of supervised learning algorithm that predicts a continuous output variable based on a single input feature. The goal is to create a linear equation that best predicts the value of the target variable.

## Equation

The simple linear regression equation is:


$
y = $\beta_0$ + $\beta_1x$ + $\epsilon$

$
where:

- \(y\) is the target variable (output)
- \(x\) is the input feature (predictor)
- \(\beta_0\) is the intercept or bias term
- \(\beta_1\) is the slope coefficient
- \(\epsilon\) is the error term (residual)

## Vectorization

To perform linear regression on a dataset, we can represent the data using vectors and matrices. Let's denote the input feature vector as **X** and the target variable vector as **y**. We can then write the linear regression equation in matrix form:

$$
y = X\beta + \epsilon
$$

where:

- **X** is an \(n \times 2\) matrix, where \(n\) is the number of samples, and each row represents a data point with a bias term (1) and the input feature value
- \(\beta\) is a \(2 \times 1\) vector, containing the intercept and slope coefficients
- \(\epsilon\) is an \(n \times 1\) vector, representing the error terms

## Important Functions and Equations

### Cost Function (Mean Squared Error - MSE)

The cost function measures the average squared difference between predicted and actual values. The goal is to minimize the cost function.

$$
J(\beta) = \frac{1}{2n} \sum{(y_i - (\beta_0 + \beta_1x_i))^2}
$$

where \(J(\beta)\) is the cost function, \(n\) is the number of samples, and \(y_i\) and \(x_i\) are the \(i\)-th target variable and input feature values, respectively.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively using the following equations:

$$
\beta_0^{new} = \beta_0^{old} - \alpha \times \frac{1}{n} \sum(-2 \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_i)))
$$

$$
\beta_1^{new} = \beta_1^{old} - \alpha \times \frac{1}{n} \sum(-2 \times x_i \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_i)))
$$

where \(\alpha\) is the learning rate, and \(\beta_0^{old}\) and \(\beta_1^{old}\) are the previous values of the intercept and slope coefficients, respectively.

### Evaluating the Fitness of the Model with a Cost Function

To evaluate the fitness of the model, we can use the cost function (MSE) to measure the average squared difference between predicted and actual values. A lower cost function value indicates a better fit.

## Solving OLS for Simple Linear Regression

Ordinary Least Squares (OLS) is a method for estimating the model parameters that minimize the sum of the squared errors. The OLS solution for simple linear regression is:

$$
\beta_1 = \frac{\sum{(x_i - \bar{x}) \times (y_i - \bar{y})}}{\sum{(x_i - \bar{x})^2}}
$$

$$
\beta_0 = \bar{y} - \beta_1 \times \bar{x}
$$

where \(\bar{x}\) and \(\bar{y}\) are the means of the input feature and target variable, respectively.

## Evaluating the Model

To evaluate the model, we can use metrics such as:

### Coefficient of Determination (R-squared)
Measures the proportion of the variance in the target variable that is predictable from the input feature.

$$
R^2 = 1 - \frac{SSE}{SST}
$$

where SSE is the sum of the squared errors, and SST is the total sum of squares.

### Mean Absolute Error (MAE)
Measures the average absolute difference between predicted and actual values.

$$
MAE = \frac{1}{n} \sum{|y_i - (\beta_0 + \beta_1x_i)|}
$$

# Multiple Linear Regression

**Multiple Linear Regression** is an extension of simple linear regression, where multiple input features are used to predict the target variable.

## Equation

The multiple linear regression equation is:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon
$$

where \(x_1, x_2, \dots, x_n\) are the input features, and \(\beta_1, \beta_2, \dots, \beta_n\) are the corresponding coefficients.

## Vectorization

The multiple linear regression equation can be written in matrix form:

$$
y = X\beta + \epsilon
$$

where:

- **X** is an \(n \times (p+1)\) matrix, where \(n\) is the number of samples, \(p\) is the number of input features, and each row represents a data point with a bias term (1) and the input feature values
- \(\beta\) is a \((p+1) \times 1\) vector, containing the intercept and slope coefficients
- \(\epsilon\) is an \(n \times 1\) vector, representing the error terms

## Important Functions and Equations

### Cost Function (Mean Squared Error - MSE)

The cost function measures the average squared difference between predicted and actual values. The goal is to minimize the cost function.

$$
J(\beta) = \frac{1}{2n} \sum{(y_i - (\beta_0 + \beta_1x_{1_i} + \beta_2x_{2_i} + \dots + \beta_nx_{n_i}))^2}
$$

where \(J(\beta)\) is the cost function, \(n\) is the number of samples, and \(y_i, x_{1_i}, x_{2_i}, \dots, x_{n_i}\) are the \(i\)-th target variable and input feature values, respectively.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively using the following equations:

$$
\beta_0^{new} = \beta_0^{old} - \alpha \times \frac{1}{n} \sum(-2 \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_{1_i} + \beta_2^{old} \times x_{2_i} + \dots + \beta_n^{old} \times x_{n_i})))
$$

$$
\beta_1^{new} = \beta_1^{old} - \alpha \times \frac{1}{n} \sum(-2 \times x_{1_i} \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_{1_i} + \beta_2^{old} \times x_{2_i} + \dots + \beta_n^{old} \times x_{n_i})))
$$

...

$$
\beta_n^{new} = \beta_n^{old} - \alpha \times \frac{1}{n} \sum(-2 \times x_{n_i} \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_{1_i} + \beta_2^{old} \times x_{2_i} + \dots + \beta_n^{old} \times x_{n_i})))
$$

where \(\alpha\) is the learning rate, and \(\beta_0^{old}, \beta_1^{old}, \beta_2^{old}, \dots, \beta_n^{old}\) are the previous values of the intercept and slope coefficients, respectively.

# Polynomial Regression

**Polynomial Regression** is a type of regression analysis where the relationship between the independent variable and the dependent variable is modeled as an nth-degree polynomial.

## Equation

The polynomial regression equation is:

$$
y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n + \epsilon
$$

where \(x\) is the input feature, and \(\beta_1, \beta_2, \dots, \beta_n\) are the corresponding coefficients.

## Vectorization

The polynomial regression equation can be written in matrix form:

$$
y = X\beta + \epsilon
$$

where:

- **X** is an \(n \times (n+1)\) matrix, where \(n\) is the number of samples, and each row represents a data point with a bias term (1) and the input feature values raised to the powers of 1, 2, \dots, \(n\)
- \(\beta\) is an \((n+1)

 \times 1\) vector, containing the intercept and slope coefficients
- \(\epsilon\) is an \(n \times 1\) vector, representing the error terms

## Important Functions and Equations

### Cost Function (Mean Squared Error - MSE)

The cost function measures the average squared difference between predicted and actual values. The goal is to minimize the cost function.

$$
J(\beta) = \frac{1}{2n} \sum{(y_i - (\beta_0 + \beta_1x_i + \beta_2x_i^2 + \dots + \beta_nx_i^n))^2}
$$

where \(J(\beta)\) is the cost function, \(n\) is the number of samples, and \(y_i\) and \(x_i\) are the \(i\)-th target variable and input feature values, respectively.

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively using the following equations:

$$
\beta_0^{new} = \beta_0^{old} - \alpha \times \frac{1}{n} \sum(-2 \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_i + \beta_2^{old} \times x_i^2 + \dots + \beta_n^{old} \times x_i^n)))
$$

$$
\beta_1^{new} = \beta_1^{old} - \alpha \times \frac{1}{n} \sum(-2 \times x_i \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_i + \beta_2^{old} \times x_i^2 + \dots + \beta_n^{old} \times x_i^n)))
$$

...

$$
\beta_n^{new} = \beta_n^{old} - \alpha \times \frac{1}{n} \sum(-2 \times x_i^n \times (y_i - (\beta_0^{old} + \beta_1^{old} \times x_i + \beta_2^{old} \times x_i^2 + \dots + \beta_n^{old} \times x_i^n)))
$$

where \(\alpha\) is the learning rate, and \(\beta_0^{old}, \beta_1^{old}, \beta_2^{old}, \dots, \beta_n^{old}\) are the previous values of the intercept and slope coefficients, respectively.

## Polynomial Feature Transformation

In polynomial regression, we transform the original input features into polynomial features. For example, for a 2nd-degree polynomial, we create the features \(x^2\) in addition to the original feature \(x\).

This transformation can be done using libraries like Scikit-learn in Python:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

## Evaluating the Model

The evaluation metrics for polynomial regression are the same as for simple and multiple linear regression, such as R-squared, Mean Absolute Error, and Mean Squared Error.

---

This concludes the guide on Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression.
```

This guide provides a clear and structured overview of linear regression techniques, from simple to polynomial. You can now use this Markdown code to format your document. Let me know if you need further customization!