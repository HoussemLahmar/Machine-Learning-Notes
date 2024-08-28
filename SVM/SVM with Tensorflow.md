#SVM with TensorFlow

### 1. Installing TensorFlow

Make sure TensorFlow is installed. You can install it using pip:

```bash
pip install tensorflow
```

### 2. Importing Required Libraries

Start by importing TensorFlow and other necessary libraries:

```python
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

### 3. Loading and Preparing Data

Load the dataset and split it into training and testing sets:

```python
# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert labels to binary for binary classification
y = (y != 0).astype(int)  # Convert to binary classification for simplicity

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 4. Building a Linear SVM with TensorFlow

#### a. Defining the Model

```python
class SVMModel(tf.keras.Model):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        return self.dense(inputs)

# Create the model
model = SVMModel()

# Define the loss function
def svm_loss(y_true, y_pred):
    return tf.reduce_mean(tf.maximum(0.0, 1.0 - y_true * y_pred))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=svm_loss)
```

#### b. Training the Model

```python
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
```

#### c. Making Predictions

```python
# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0).astype(int).flatten()

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Linear SVM Accuracy: {accuracy:.2f}")
```

### 5. Nonlinear SVM with Kernel Trick

For nonlinear SVMs, TensorFlow doesn't have direct support for kernel methods like polynomial or RBF kernels. You can manually implement a kernel function and add it to your model.

#### a. Creating Kernel Features

```python
def rbf_kernel(X1, X2, gamma=0.5):
    """ Radial Basis Function (RBF) kernel """
    sq_dists = tf.reduce_sum(tf.square(X1[:, tf.newaxis] - X2), axis=2)
    return tf.exp(-gamma * sq_dists)
    
# Transform the input features using the RBF kernel
X_train_kernel = rbf_kernel(X_train, X_train)
X_test_kernel = rbf_kernel(X_test, X_train)
```

#### b. Building the Model with Kernel Features

```python
class KernelSVMModel(tf.keras.Model):
    def __init__(self, num_features):
        super(KernelSVMModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        return self.dense(inputs)

# Create the model
model_kernel = KernelSVMModel(num_features=X_train_kernel.shape[1])

# Define the loss function
def svm_loss_kernel(y_true, y_pred):
    return tf.reduce_mean(tf.maximum(0.0, 1.0 - y_true * y_pred))

# Compile the model
model_kernel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=svm_loss_kernel)

# Train the model
model_kernel.fit(X_train_kernel, y_train, epochs=100, batch_size=32, verbose=1)
```

#### c. Making Predictions with Kernel Features

```python
# Make predictions
y_pred_kernel = model_kernel.predict(X_test_kernel)
y_pred_binary_kernel = (y_pred_kernel > 0).astype(int).flatten()

# Evaluate accuracy
accuracy_kernel = accuracy_score(y_test, y_pred_binary_kernel)
print(f"Kernel SVM Accuracy: {accuracy_kernel:.2f}")
```

### Summary

1. **Install TensorFlow** and import necessary libraries.
2. **Load and preprocess** your dataset.
3. **Create a linear SVM model** using TensorFlow's `tf.keras.Model`.
4. **Train and evaluate** the linear SVM.
5. **Implement kernel methods** manually and use them for nonlinear SVM classification.
