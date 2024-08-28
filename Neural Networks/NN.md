# Neural Networks Course Notes

## 1. Introduction

Neural networks are a subset of machine learning algorithms inspired by the human brain's structure and function. They are composed of interconnected layers of neurons that process data and learn patterns through training. Neural networks are widely used for tasks such as image recognition, natural language processing, and time series forecasting.

### Key Concepts
- **Neuron:** The basic unit of a neural network that receives inputs, applies weights, and passes the result through an activation function.
- **Layer:** A collection of neurons. Neural networks are typically organized into input, hidden, and output layers.
- **Activation Function:** A function applied to the neuron's output to introduce non-linearity. Common functions include Sigmoid, Tanh, and ReLU.
- **Loss Function:** A function that measures the error between the predicted output and the actual target. Examples include Mean Squared Error (MSE) and Cross-Entropy Loss.
- **Optimizer:** An algorithm used to minimize the loss function by adjusting the network's weights. Examples include Gradient Descent and Adam.

## 2. Building a Perceptron

The perceptron is the simplest type of neural network, consisting of a single layer of neurons. It is used for binary classification tasks.

### Mathematical Model

A perceptron calculates a weighted sum of the inputs and applies an activation function:

$$
y = \text{activation}(w \cdot x + b)
$$

where:
- \( w \) is the weight vector,
- \( x \) is the input vector,
- \( b \) is the bias term,
- \( \text{activation} \) is the step function for binary classification.

### Algorithm
1. **Initialize Weights and Bias:** Randomly initialize the weights and bias.
2. **Forward Pass:** Compute the weighted sum of inputs and apply the activation function.
3. **Update Weights:** Adjust weights based on the error using the learning rule:

$$
w = w + \alpha (t - y) x
$$

where:
- \( \alpha \) is the learning rate,
- \( t \) is the target output,
- \( y \) is the predicted output.

## 3. Building a Single Layer Neural Network

A single-layer neural network, also known as a single-layer perceptron, can classify linearly separable data.

### Mathematical Model

The network has an input layer and an output layer. The output is computed as:

$$
y = \text{activation}(W \cdot X + b)
$$

where:
- \( W \) is the weight matrix,
- \( X \) is the input matrix,
- \( b \) is the bias vector.

### Algorithm
1. **Initialize Weights and Biases:** Randomly initialize.
2. **Forward Pass:** Compute the output by applying the activation function to the weighted sum.
3. **Loss Calculation:** Use a loss function to measure the error.
4. **Backpropagation:** Update weights and biases by calculating gradients and applying the optimizer.

## 4. Building a Deep Neural Network

Deep Neural Networks (DNNs) consist of multiple hidden layers, allowing them to learn complex patterns.

### Mathematical Model

For a network with \( L \) layers, the output \( y \) is:

$$
y = \text{activation}(W_L \cdot \text{activation}(W_{L-1} \cdot \ldots \cdot \text{activation}(W_1 \cdot X + b_1) + b_{L-1}) + b_L)
$$

### Algorithm
1. **Initialization:** Randomly initialize weights and biases.
2. **Forward Pass:** Pass data through each layer, applying the activation function.
3. **Loss Calculation:** Compute the loss using a suitable loss function.
4. **Backpropagation:** Compute gradients of the loss function with respect to weights and biases, then update them using an optimizer.
5. **Iteration:** Repeat until convergence or a stopping criterion is met.

### Example Architecture
- **Input Layer:** Number of neurons = Number of features.
- **Hidden Layers:** Multiple layers with varying neurons and activation functions.
- **Output Layer:** Number of neurons = Number of classes (for classification) or 1 (for regression).

## 5. Building a Recurrent Neural Network for Sequential Data Analysis

Recurrent Neural Networks (RNNs) are designed for sequential data, where current input depends on previous inputs.

### Mathematical Model

The RNN computes the hidden state \( h_t \) and output \( y_t \) as:

$$
h_t = \text{activation}(W_h \cdot h_{t-1} + U \cdot x_t + b_h)
$$

$$
y_t = W_y \cdot h_t + b_y
$$

where:
- \( W_h \) is the weight matrix for the hidden state,
- \( U \) is the weight matrix for the input,
- \( b_h \) is the bias for the hidden state,
- \( W_y \) is the weight matrix for the output,
- \( b_y \) is the output bias.

### Algorithm
1. **Initialization:** Initialize weights and biases.
2. **Forward Pass:** Compute hidden states and outputs sequentially.
3. **Loss Calculation:** Compute the loss using a suitable function.
4. **Backpropagation Through Time (BPTT):** Compute gradients for each time step and update weights.
5. **Iteration:** Repeat until convergence.

### Variants
- **Long Short-Term Memory (LSTM):** Addresses long-term dependencies and vanishing gradient problem.
- **Gated Recurrent Units (GRU):** A simplified version of LSTMs with fewer gates.

## 6. Visualizing the Characters in an Optical Character Recognition Database

Optical Character Recognition (OCR) databases often contain images of characters and their labels. Visualization helps understand the dataset's structure and quality.

### Example Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Load OCR dataset (example)
images, labels = load_ocr_dataset()

# Plot a sample image
plt.imshow(images[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()
```

## 7. Building an Optical Character Recognizer Using Neural Networks

Optical Character Recognition (OCR) involves classifying images of text into corresponding characters.

### Steps
1. **Data Preparation:**
   - Load and preprocess images (resize, normalize).
   - Split dataset into training and test sets.
2. **Model Architecture:**
   - **Convolutional Neural Network (CNN):** Commonly used for image classification tasks.
   
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

     model = Sequential([
         Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
         MaxPooling2D((2, 2)),
         Flatten(),
         Dense(128, activation='relu'),
         Dense(10, activation='softmax')
     ])
     ```

3. **Training:**
   - Compile the model with a suitable optimizer and loss function.
   - Train using the training set and validate on the test set.
   
     ```python
     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
     ```

4. **Evaluation:**
   - Evaluate the model on the test set and adjust parameters if necessary.

5. **Inference:**
   - Use the trained model to make predictions on new images.

## 8. Summary

Neural networks are a versatile tool in machine learning, capable of handling a wide range of tasks from classification to sequence analysis. Key topics include:

- **Perceptrons** for basic binary classification.
- **Single-Layer Networks** for simple classification tasks.
- **Deep Neural Networks** for complex pattern recognition.
- **Recurrent Neural Networks** for sequential data.
- **OCR** for character recognition using CNNs.

