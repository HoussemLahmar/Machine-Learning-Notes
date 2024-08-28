# Logistic Regression with TensorFlow 

## 1. Logistic Regression

### Definition

Logistic Regression is a supervised learning algorithm used for binary classification problems. It predicts the probability of an event occurring based on a set of input variables.

### Mathematical Representation

In a binary classification problem, the logistic regression model predicts the probability \( p \) of an event occurring (e.g., 1) or not occurring (e.g., 0). The model is represented as:

$$ p(y=1|x) = \frac{1}{1 + e^{-z}} $$

where \( z = w^T x + b \), \( w \) is the weight vector, \( b \) is the bias term, and \( e \) is the base of the natural logarithm.

## 2. Using TensorFlow for Logistic Regression

### Importing Libraries

```python
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Loading Data

For demonstration, we'll use the Iris dataset:

```python
data = load_iris()
X = data.data
y = data.target

# Since Iris dataset is multi-class, we use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]
```

### Preprocessing Data

Split the data into training and testing sets, and standardize it:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Building the Model

Create a logistic regression model using TensorFlow:

```python
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Compiling the Model

Compile the model with an appropriate loss function and optimizer:

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Training the Model

Train the model on the training data:

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)
```

### Making Predictions

Predict the target values for the test set:

```python
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
```

### Evaluating the Model

Evaluate the model using accuracy, confusion matrix, and classification report:

```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
```

## 3. Performance Metrics

### Accuracy

Accuracy measures the proportion of correctly classified instances out of all instances:

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

### Precision and Recall

- **Precision**: Proportion of true positives among all predicted positive instances.

  $$ \text{Precision} = \frac{TP}{TP + FP} $$

- **Recall**: Proportion of true positives among all actual positive instances.

  $$ \text{Recall} = \frac{TP}{TP + FN} $$

### F1 Score

The F1 score is the harmonic mean of precision and recall:

$$ \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall} $$

### ROC AUC

The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate at different thresholds. The Area Under the Curve (AUC) represents the model's ability to distinguish between positive and negative classes.

## 4. How to Approach Classification Problems

### Steps

1. **Data Preprocessing**: Handle missing values, normalize/scale features, and encode categorical variables.
2. **Feature Engineering**: Extract relevant features from the data.
3. **Model Selection**: Choose a suitable classification algorithm based on the problem and data.
4. **Hyperparameter Tuning**: Optimize model parameters for better performance.
5. **Model Evaluation**: Use metrics such as accuracy, precision, recall, F1, and ROC AUC to evaluate the model.

## 5. Datasets

### Examples

- **Predicting Insurance**
  - **Dataset**: Insurance claims data
  - **Task**: Predict whether a claim is approved or rejected
  - **Features**: Claim amount, age, location, etc.

- **Spam Filtering**
  - **Dataset**: Email data
  - **Task**: Classify emails as spam or not spam
  - **Features**: Email content, sender, recipient, etc.

- **Digit Classification**
  - **Dataset**: Handwritten digit images
  - **Task**: Classify images as digits (0-9)
  - **Features**: Pixel values, shape, size, etc.

- **Titanic Dataset**
  - **Dataset**: Passenger data from the Titanic
  - **Task**: Predict whether a passenger survived or not
  - **Features**: Age, sex, class, etc.

## 6. Summary

### Overview

Logistic regression is a powerful algorithm for binary classification problems. TensorFlow provides a flexible and efficient way to implement logistic regression models. By understanding the mathematical representation, performance metrics, and how to use TensorFlow, you can build and evaluate logistic regression models effectively for various datasets.
