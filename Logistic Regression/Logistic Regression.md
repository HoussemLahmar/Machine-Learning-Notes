# Logistic Regression 

## 1. Logistic Regression

### Definition

Logistic Regression is a type of supervised learning algorithm used for binary classification problems. It predicts the probability of an event occurring based on a set of input variables.

### Mathematical Representation

Let's consider a binary classification problem where we want to predict the probability of an event occurring (e.g., 1) or not occurring (e.g., 0). We have a set of input features \( X = [x_1, x_2, \ldots, x_n] \) and a target variable \( y \). The logistic regression model can be represented as:

$$ p(y=1|x) = \frac{1}{1 + e^{-z}} $$

where \( z = w^T x + b \), \( w \) is the weight vector, \( b \) is the bias term, and \( e \) is the base of the natural logarithm.

## 2. Binary Classification

### Definition

Binary classification is a type of classification problem where we have two classes or labels. The goal is to predict the probability of an instance belonging to one of the two classes.

### Examples

- Spam vs. Not Spam emails
- Cancer vs. Not Cancer diagnosis

## 3. Performance Matrix

### Definition

A performance matrix is a table used to evaluate the performance of a classification model. It contains the following metrics:

|                     | Predicted Positive | Predicted Negative |
|---------------------|---------------------|---------------------|
| **Actual Positive** | True Positives (TP) | False Negatives (FN)|
| **Actual Negative** | False Positives (FP)| True Negatives (TN) |

## 4. Accuracy

### Formula

Accuracy is the proportion of correctly classified instances out of all instances in the dataset.

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

## 5. Precision and Recall

### Formulas

- **Precision** is the proportion of true positives among all predicted positive instances.

  $$ \text{Precision} = \frac{TP}{TP + FP} $$

- **Recall** is the proportion of true positives among all actual positive instances.

  $$ \text{Recall} = \frac{TP}{TP + FN} $$

## 6. F1 Measure

### Formula

The F1 measure is the harmonic mean of precision and recall.

$$ \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall} $$

## 7. ROC AUC

### Definition

The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate at different thresholds. The Area Under the Curve (AUC) represents the model's ability to distinguish between positive and negative classes.

## 8. How to Approach Classification Problems

### Steps

1. **Data Preprocessing**: Handle missing values, normalize/scale features, and encode categorical variables.
2. **Feature Engineering**: Extract relevant features from the data.
3. **Model Selection**: Choose a suitable classification algorithm based on the problem and data.
4. **Hyperparameter Tuning**: Optimize model parameters for better performance.
5. **Model Evaluation**: Use metrics such as accuracy, precision, recall, F1, and ROC AUC to evaluate the model.

## 9. Datasets

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

## 10. Summary

### Overview

Logistic regression is a powerful algorithm for binary classification problems. Understanding the mathematical representation, performance metrics, and approaches to classification problems is crucial for building accurate models. By applying these concepts to various datasets, we can develop robust models that can make accurate predictions in real-world scenarios.

