# Scikit-Learn Cheat Sheet

## Basic Scikit-Learn Operations

### Import Scikit-Learn
```python
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Loading Datasets
```python
from sklearn.datasets import load_iris, load_boston

# Load datasets
iris = load_iris()
boston = load_boston()

# Data and target
X, y = iris.data, iris.target
```

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Building Machine Learning Models

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Building Neural Networks

### Multi-layer Perceptron (MLP)
```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Building Convolutional Neural Networks (CNNs)

**Note:** Scikit-Learn does not support CNNs directly. For CNNs, use libraries such as TensorFlow or PyTorch.

## Model Evaluation

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Score:", scores.mean())
```

### Grid Search for Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

# Grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
```

## Workflow for Scikit-Learn Projects

1. **Define the Problem:**
   - Understand the problem and data requirements.

2. **Prepare the Data:**
   - Load and explore the dataset.
   - Handle missing values and outliers.
   - Split the data into training and test sets.

3. **Feature Engineering:**
   - Scale features if needed.
   - Encode categorical variables.

4. **Choose a Model:**
   - Select an appropriate model based on the problem (e.g., classification, regression).

5. **Train the Model:**
   - Fit the model to the training data.

6. **Evaluate the Model:**
   - Use metrics such as accuracy, precision, recall, F1-score, or mean squared error.
   - Perform cross-validation if necessary.

7. **Hyperparameter Tuning:**
   - Use techniques like Grid Search or Random Search to optimize model parameters.

8. **Make Predictions:**
   - Use the trained model to make predictions on new data.

9. **Deploy the Model:**
   - Save the model and deploy it for real-world use.

10. **Monitor and Maintain:**
    - Track model performance and update it as needed.

