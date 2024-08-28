# Naive Bayes with Scikit-Learn

Scikit-Learn is a powerful machine learning library in Python that provides easy-to-use implementations of various machine learning algorithms, including Naive Bayes.

## 1. Importing Libraries

First, you need to import the necessary libraries:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## 2. Loading the Data

For demonstration, let’s use a sample dataset. Here’s how to load data using Pandas:

```python
# Load dataset (example: Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
```

## 3. Splitting the Data

Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4. Training Naive Bayes Models

Scikit-Learn provides different types of Naive Bayes models:

### Gaussian Naive Bayes

```python
# Initialize and train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```

### Multinomial Naive Bayes

```python
# Initialize and train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
```

### Bernoulli Naive Bayes

```python
# Initialize and train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
```

## 5. Making Predictions

Use the trained model to make predictions on the test data:

```python
# Predict using Gaussian Naive Bayes
y_pred_gnb = gnb.predict(X_test)

# Predict using Multinomial Naive Bayes
y_pred_mnb = mnb.predict(X_test)

# Predict using Bernoulli Naive Bayes
y_pred_bnb = bnb.predict(X_test)
```

## 6. Evaluating the Model

Evaluate the performance of each model using metrics like accuracy, confusion matrix, and classification report:

### Gaussian Naive Bayes

```python
print("Gaussian Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))
print("Classification Report:\n", classification_report(y_test, y_pred_gnb))
```

### Multinomial Naive Bayes

```python
print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mnb))
print("Classification Report:\n", classification_report(y_test, y_pred_mnb))
```

### Bernoulli Naive Bayes

```python
print("Bernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bnb))
print("Classification Report:\n", classification_report(y_test, y_pred_bnb))
```

## 7. Summary

Scikit-Learn offers a straightforward way to implement and evaluate Naive Bayes classifiers. Depending on the nature of your data (e.g., continuous vs. discrete features), you can choose among Gaussian, Multinomial, or Bernoulli Naive Bayes models.

### Advantages
- Simple and easy to implement
- Fast to train and predict
- Works well with large datasets

### Disadvantages
- Assumes feature independence
- May perform poorly with correlated features

