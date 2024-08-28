# Naive Bayes

## 1. Naive Bayes Mathematical Concept

Naive Bayes is a family of probabilistic machine learning models based on Bayes' theorem, which describes the probability of an event given prior knowledge of conditions that might be related to the event.

### Bayes' Theorem

Bayes' theorem is a mathematical formula that describes the probability of an event \(A\) given event \(B\):

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

where:
- \(P(A|B)\) is the posterior probability of event \(A\) given event \(B\)
- \(P(B|A)\) is the likelihood of event \(B\) given event \(A\)
- \(P(A)\) is the prior probability of event \(A\)
- \(P(B)\) is the prior probability of event \(B\)

## 2. Bayes' Theorem

Bayes' theorem allows us to update the probability of an event based on new evidence.

### Example

Suppose we want to determine the probability of a person having a disease (D) given a positive test result (T). We know the following probabilities:
- \(P(D) = 0.01\) (prior probability of the disease)
- \(P(T|D) = 0.9\) (likelihood of a positive test result given the disease)
- \(P(T) = 0.1\) (prior probability of a positive test result)

Using Bayes' theorem, we can calculate the posterior probability of the disease given a positive test result:

$$
P(D|T) = \frac{P(T|D) \cdot P(D)}{P(T)} = \frac{0.9 \cdot 0.01}{0.1} = 0.09
$$

## 3. Generative and Discriminative Models

Machine learning models can be classified into two categories: generative and discriminative models.

### Generative Models

Generative models model the joint probability distribution of the input data and the target variable. They can generate new data samples that are similar to the training data.

### Discriminative Models

Discriminative models model the conditional probability distribution of the target variable given the input data. They can predict the target variable for new input data.

## 4. Naive Bayes

Naive Bayes is a family of generative models based on Bayes' theorem. They are called "naive" because they make simplifying assumptions about the data.

### Naive Bayes Algorithm

The Naive Bayes algorithm can be summarized as follows:
1. **Calculate the prior probabilities**: Calculate the prior probabilities of each class label.
2. **Calculate the likelihoods**: Calculate the likelihood of each feature given each class label.
3. **Calculate the posterior probabilities**: Calculate the posterior probabilities of each class label given the input data using Bayes' theorem.
4. **Make predictions**: Make predictions based on the posterior probabilities.

## 5. Assumptions of Naive Bayes

Naive Bayes makes the following assumptions about the data:
- **Independence**: The features are independent of each other given the class label.
- **Identical distribution**: The features have identical distributions given the class label.
- **No missing values**: There are no missing values in the data.

## 6. Solving Dataset with Problems

Let's consider a dataset with two features (X1 and X2) and two class labels (Y1 and Y2). We want to use Naive Bayes to classify new data points.

### Dataset

| X1 | X2 | Y  |
|----|----|----|
| 1  | 2  | Y1 |
| 2  | 3  | Y1 |
| 3  | 1  | Y2 |
| 4  | 4  | Y2 |
| ...| ...| ...|

### Solution

We can use Naive Bayes to classify new data points by calculating the posterior probabilities of each class label given the input data.

Let's say we want to classify a new data point with \(X1 = 2.5\) and \(X2 = 3.5\). We can calculate the posterior probabilities as follows:

$$
P(Y1|X1=2.5, X2=3.5) = \frac{P(X1=2.5|Y1) \cdot P(X2=3.5|Y1) \cdot P(Y1)}{P(X1=2.5, X2=3.5)}
$$

$$
P(Y2|X1=2.5, X2=3.5) = \frac{P(X1=2.5|Y2) \cdot P(X2=3.5|Y2) \cdot P(Y2)}{P(X1=2.5, X2=3.5)}
$$

Assume we get the following values:
- \(P(X1=2.5|Y1) = 0.4\)
- \(P(X2=3.5|Y1) = 0.6\)
- \(P(Y1) = 0.5\)
- \(P(X1=2.5|Y2) = 0.3\)
- \(P(X2=3.5|Y2) = 0.7\)
- \(P(Y2) = 0.5\)

We can then calculate the posterior probabilities:

$$
P(Y1|X1=2.5, X2=3.5) = \frac{0.4 \cdot 0.6 \cdot 0.5}{P(X1=2.5, X2=3.5)} = 0.12
$$

$$
P(Y2|X1=2.5, X2=3.5) = \frac{0.3 \cdot 0.7 \cdot 0.5}{P(X1=2.5, X2=3.5)} = 0.21
$$

Since \(P(Y2|X1=2.5, X2=3.5) > P(Y1|X1=2.5, X2=3.5)\), we classify the new data point as Y2.

## 7. Summary

Naive Bayes is a family of generative models based on Bayes' theorem. They are simple and effective, but make some simplifying assumptions about the data. Naive Bayes can be used for classification tasks and is particularly useful when the features are independent and identically distributed.

### Advantages of Naive Bayes
- Simple to implement
- Fast computation
- Handles high-dimensional data
- Can handle missing values

### Disadvantages of Naive Bayes
- Assumes independence of features
- Assumes identical distribution of features
- Can be sensitive to outliers

### Real-World Applications of Naive Bayes
- Text classification
- Sentiment analysis
- Image classification
- Bioinformatics

