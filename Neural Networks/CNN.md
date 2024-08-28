# Convolutional Neural Networks (CNNs) Course Notes

## 1. Introducing the CNN

Convolutional Neural Networks (CNNs) are a class of deep neural networks specifically designed to process structured grid data, such as images. CNNs use convolutional layers that apply filters to local regions of the input to capture spatial hierarchies.

### Key Concepts
- **Local Receptive Fields:** Each neuron in the convolutional layer is connected to a small region of the input.
- **Shared Weights:** The same filter (weights) is applied across the entire input.
- **Spatial Hierarchy:** Captures patterns like edges, textures, and shapes by stacking layers.

## 2. Understanding the ConvNet Topology

CNNs typically consist of several key layers:
- **Input Layer:** Takes the raw image data.
- **Convolutional Layers:** Apply convolutional filters to extract features.
- **Activation Layers:** Apply non-linear activation functions (e.g., ReLU).
- **Pooling Layers:** Reduce the spatial dimensions (e.g., max pooling).
- **Fully Connected Layers:** Output the final class predictions.

### CNN Architecture

```
Input Image -> Convolution Layer -> Activation Function -> Pooling Layer -> Fully Connected Layer -> Output
```

## 3. Understanding Convolution Layers

Convolution layers are the core building blocks of CNNs. They apply filters (kernels) to the input to produce feature maps.

### Mathematical Model

For a 2D convolution:

$$
S(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(i+m, j+n) \cdot K(m, n) + B
$$

where:
- \( S(i, j) \) is the output feature map at position \((i, j)\),
- \( I \) is the input image,
- \( K \) is the convolution kernel,
- \( B \) is the bias term,
- \( M \) and \( N \) are the dimensions of the kernel.

### Example Filter Application

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Define a convolutional layer
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
```

## 4. Understanding Pooling Layers

Pooling layers reduce the spatial dimensions of the feature maps, helping to decrease computational load and control overfitting.

### Mathematical Model

For max pooling:

$$
P(i, j) = \max_{(m, n) \in \text{pool\_region}} F(i+m, j+n)
$$

where:
- \( P(i, j) \) is the pooled feature map value,
- \( F \) is the feature map,
- The pooling region is defined by the pooling window size.

### Example Pooling Operation

```python
from tensorflow.keras.layers import MaxPooling2D

# Define a max pooling layer
pool_layer = MaxPooling2D(pool_size=(2, 2))
```

## 5. Training a ConvNet

Training a CNN involves feeding labeled data, optimizing the network parameters, and evaluating its performance.

### Training Steps
1. **Data Preparation:** Preprocess images (e.g., normalization).
2. **Model Definition:** Build the CNN architecture.
3. **Compilation:** Choose optimizer, loss function, and metrics.
4. **Training:** Fit the model on training data.
5. **Evaluation:** Test the model on unseen data.

### Example Training Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

## 6. Putting It All Together

Combining the convolutional, activation, pooling, and fully connected layers results in a complete CNN architecture that can learn and classify complex patterns in image data.

### Full CNN Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the complete CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## 7. Applying a CNN

CNNs are widely used for tasks such as image classification, object detection, and semantic segmentation.

### Example Application: Image Classification

1. **Load Data:** Use datasets like MNIST, CIFAR-10.
2. **Train Model:** Use the previously defined CNN architecture.
3. **Evaluate:** Measure accuracy on test data.
4. **Deploy:** Use the model for predicting new images.

### Example Application Code

```python
# Predict on new images
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
```

## 8. Summary

Convolutional Neural Networks (CNNs) are powerful for processing grid-like data. Key points include:

- **CNN Architecture:** Comprised of convolutional layers, activation functions, pooling layers, and fully connected layers.
- **Convolution and Pooling:** Fundamental operations that help capture spatial hierarchies and reduce dimensionality.
- **Training:** Involves defining, compiling, and fitting the model.
- **Applications:** Versatile in tasks such as image classification and object detection.

