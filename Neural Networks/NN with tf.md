# Neural Networks with TensorFlow Course Notes

## 1. Introduction

Neural networks, inspired by the structure and function of the human brain, are used for a variety of tasks such as image recognition, natural language processing, and more. TensorFlow is an open-source library developed by Google for machine learning and deep learning applications, providing tools for building, training, and deploying neural networks.

### Key Concepts
- **TensorFlow:** An open-source library for numerical computation and machine learning.
- **Tensor:** A multi-dimensional array used as the basic data structure in TensorFlow.
- **Model:** A neural network or machine learning algorithm that learns from data.
- **Keras:** An API within TensorFlow that provides a high-level interface for building and training models.

## 2. Building a Perceptron with TensorFlow

A perceptron is the simplest neural network model, consisting of a single layer used for binary classification.

### Mathematical Model

The perceptron calculates:

$$
y = \text{activation}(w \cdot x + b)
$$

where:
- \( w \) is the weight vector,
- \( x \) is the input vector,
- \( b \) is the bias term,
- \( \text{activation} \) is typically the step function.

### TensorFlow Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR function

# Define the model
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2,))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000)
```

## 3. Building a Single Layer Neural Network with TensorFlow

A single-layer neural network (or single-layer perceptron) is used for linearly separable data.

### Mathematical Model

For a single-layer network:

$$
y = \text{activation}(W \cdot X + b)
$$

where:
- \( W \) is the weight matrix,
- \( X \) is the input matrix,
- \( b \) is the bias vector.

### TensorFlow Implementation

```python
# Define the model
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2,))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000)
```

## 4. Building a Deep Neural Network with TensorFlow

Deep Neural Networks (DNNs) consist of multiple hidden layers, allowing for complex pattern recognition.

### Mathematical Model

For a deep network:

$$
y = \text{activation}(W_L \cdot \text{activation}(W_{L-1} \cdot \ldots \cdot \text{activation}(W_1 \cdot X + b_1) + b_{L-1}) + b_L)
$$

### TensorFlow Implementation

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate sample data
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=1000)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

## 5. Building a Recurrent Neural Network for Sequential Data Analysis with TensorFlow

Recurrent Neural Networks (RNNs) are designed for sequential data, where the current input depends on previous inputs.

### Mathematical Model

The RNN computes:

$$
h_t = \text{activation}(W_h \cdot h_{t-1} + U \cdot x_t + b_h)
$$

$$
y_t = W_y \cdot h_t + b_y
$$

### TensorFlow Implementation

```python
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# Define the model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(None, 10)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate sample sequential data
X_train = np.random.rand(1000, 10, 10)  # 1000 sequences of length 10 with 10 features
y_train = np.random.randint(2, size=1000)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

## 6. Visualizing the Characters in an Optical Character Recognition Database

Visualizing data helps understand and prepare datasets for training.

### Example Visualization

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset (example)
mnist = tf.keras.datasets.mnist
(train_images, train_labels), _ = mnist.load_data()

# Plot a sample image
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()
```

## 7. Building an Optical Character Recognizer Using Neural Networks with TensorFlow

Optical Character Recognition (OCR) involves classifying images of text into corresponding characters.

### Steps

1. **Data Preparation:**
   - Load and preprocess images (e.g., normalization).
   - Split dataset into training and test sets.

2. **Model Architecture:**
   - **Convolutional Neural Network (CNN):** Effective for image classification tasks.
   
     ```python
     from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
     from tensorflow.keras.models import Sequential

     # Define the model
     model = Sequential([
         Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
         MaxPooling2D((2, 2)),
         Flatten(),
         Dense(128, activation='relu'),
         Dense(10, activation='softmax')
     ])

     # Compile the model
     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

     # Load MNIST dataset
     mnist = tf.keras.datasets.mnist
     (X_train, y_train), (X_test, y_test) = mnist.load_data()
     X_train = X_train[..., np.newaxis] / 255.0
     X_test = X_test[..., np.newaxis] / 255.0

     # Train the model
     model.fit(X_train, y_train, epochs=5, validation_split=0.2)
     ```

3. **Evaluation:**
   - Evaluate the model on test data and adjust parameters if necessary.

4. **Inference:**
   - Use the trained model to predict new images.

## 8. Summary

Neural networks, powered by TensorFlow, provide robust tools for various machine learning tasks. Key topics include:

- **Perceptrons** for binary classification.
- **Single-Layer Networks** for simple tasks.
- **Deep Neural Networks** for complex patterns.
- **Recurrent Neural Networks** for sequential data.
- **Optical Character Recognition** using CNNs for image classification.
