### Linear Regression 

**Simple Linear Regression:**
Simple Linear Regression is a fundamental supervised learning algorithm that predicts a continuous output variable based on a single input feature. The relationship between the input and output is modeled as a straight line, which can be expressed as:

\[ y = \beta_0 + \beta_1x + \epsilon \]

- \( y \): Target variable (output)
- \( x \): Input feature (predictor)
- \( \beta_0 \): Intercept (bias term)
- \( \beta_1 \): Slope coefficient (weight)
- \( \epsilon \): Error term (residual)

**Vectorization:**
To efficiently perform linear regression on a dataset, vectorization is employed. The regression equation in matrix form is:

\[ y = X\beta + \epsilon \]

- \( X \): An \( n \times 2 \) matrix (where \( n \) is the number of samples), including the bias term (1) and the input feature values.
- \( \beta \): A \( 2 \times 1 \) vector containing the intercept and slope coefficients.
- \( \epsilon \): An \( n \times 1 \) vector representing the error terms.

### Important Functions and Equations

**Cost Function (Mean Squared Error - MSE):**
The cost function for linear regression is the Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values:

\[ J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - (\beta_0 + \beta_1x_i)\right)^2 \]

Here, \( J(\beta) \) is the cost function, \( n \) is the number of samples, and \( y_i \) and \( x_i \) represent the \( i \)-th target variable and input feature value.

**Gradient Descent:**
Gradient descent is an iterative optimization technique used to minimize the cost function by updating the model parameters:

- **For intercept (\( \beta_0 \)):**

  \[ \beta_0^{new} = \beta_0^{old} - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} \left(-2 \cdot (y_i - (\beta_0^{old} + \beta_1^{old} \cdot x_i))\right) \]

- **For slope (\( \beta_1 \)):**

  \[ \beta_1^{new} = \beta_1^{old} - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} \left(-2 \cdot x_i \cdot (y_i - (\beta_0^{old} + \beta_1^{old} \cdot x_i))\right) \]

\( \alpha \) is the learning rate, which controls the step size during parameter updates.

### Evaluating the Model

**Coefficient of Determination (R-squared):**
R-squared measures the proportion of variance in the target variable that is predictable from the input feature:

\[ R^2 = 1 - \frac{SSE}{SST} \]

Where \( SSE \) is the sum of squared errors, and \( SST \) is the total sum of squares.

**Mean Absolute Error (MAE):**
MAE measures the average absolute difference between predicted and actual values:

\[ MAE = \frac{1}{n} \sum_{i=1}^{n} \left|y_i - (\beta_0 + \beta_1x_i)\right| \]

### Multiple Linear Regression

**Equation:**
Multiple Linear Regression extends simple linear regression by considering multiple input features:

\[ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon \]

**Vectorization:**
In matrix form, the equation becomes:

\[ y = X\beta + \epsilon \]

- \( X \): An \( n \times (p+1) \) matrix, where \( p \) is the number of features.

**Gradient Descent Updates:**
For each coefficient:

- \( \beta_j^{new} = \beta_j^{old} - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} \left(-2 \cdot x_{ij} \cdot (y_i - (\beta_0^{old} + \dots + \beta_j^{old} \cdot x_{ij}))\right) \)

### Polynomial Regression

Polynomial Regression models the relationship between the independent variable and the dependent variable as an \( n \)-th degree polynomial:

\[ y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n + \epsilon \]

In matrix form:

\[ y = X\beta + \epsilon \]

### Applying Linear Regression

Linear regression can be used for various applications, including predicting continuous outcomes, analyzing relationships between variables, and identifying patterns. Before applying linear regression, it's essential to explore and visualize the data, using scatter plots or calculating the correlation coefficient to understand the relationships.

### Conclusion

Linear regression is a foundational tool in machine learning for predicting continuous outcomes and analyzing relationships. Mastery of the underlying equations, cost functions, and optimization techniques, along with data exploration, can significantly enhance the performance and insights of your models.