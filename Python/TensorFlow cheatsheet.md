# TensorFlow Cheat Sheet

## Basic TensorFlow Operations

### Import TensorFlow
```python
import tensorflow as tf
```

### Create Tensors
```python
# Constant tensor
a = tf.constant([1, 2, 3])

# Variable tensor
b = tf.Variable([4, 5, 6])
```

### Tensor Operations
```python
# Addition
c = tf.add(a, b)

# Multiplication
d = tf.multiply(a, b)

# Matrix Multiplication
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
```

### Reshaping
```python
tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
reshaped_tensor = tf.reshape(tensor, (4, 2))
```

## Building Machine Learning Models

### Linear Regression
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Model
model = Sequential([Dense(1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, y, epochs=1000)

# Predict
model.predict([5])
```

### Logistic Regression
```python
# Data
X = np.array([[0], [1], [2], [3]], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)

# Model
model = Sequential([Dense(1, activation='sigmoid', input_shape=[1])])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000)

# Predict
model.predict([2.5])
```

## Building Neural Networks

### Single Layer Neural Network
```python
# Data
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Model
model = Sequential([Dense(1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, y, epochs=1000)
```

### Multi-Layer Neural Network
```python
# Model
model = Sequential([
    Dense(64, activation='relu', input_shape=[1]),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=1000)
```

## Building Convolutional Neural Networks (CNNs)

### CNN Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Model
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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Training CNN
```python
# Load and preprocess dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

## TensorFlow Project Workflow

1. **Define the Problem:** Understand the problem and data.
2. **Prepare the Data:**
   - Collect and clean data.
   - Split into training, validation, and test sets.
3. **Choose a Model Architecture:**
   - Select model type (e.g., CNN for images).
4. **Build the Model:**
   - Define model using TensorFlow/Keras.
5. **Compile the Model:**
   - Choose optimizer, loss function, and metrics.
6. **Train the Model:**
   - Fit model to training data.
   - Monitor performance on validation data.
7. **Evaluate the Model:**
   - Test performance on test data.
8. **Deploy the Model:**
   - Save model and deploy.
   - Make predictions on new data.
9. **Monitor and Maintain:**
   - Track performance.
   - Update as necessary.

## Existing Models in TensorFlow

### Image Classification
- **InceptionV3**
- **ResNet**
- **VGG16/19**
- **MobileNet**

### Object Detection
- **YOLO (You Only Look Once)**
- **SSD (Single Shot MultiBox Detector)**

### Natural Language Processing
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **GPT (Generative Pre-trained Transformer)**

### Generative Models
- **GANs (Generative Adversarial Networks)**
- **VAEs (Variational Autoencoders)**

### Using Pre-trained Models
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained model
model = VGG16(weights='imagenet')

# Load and preprocess image
img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict
predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Predictions:", decoded_predictions)
```

