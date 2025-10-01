import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Flatten, Conv2D
from keras.datasets import mnist
from keras import Sequential
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Flatten, Conv2D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras import Sequential
import matplotlib.pyplot as plt
import random
import math
import re
import os
import time
import logging
import sys
import datetime


# Suppress TensorFlow logging for clarity

# Suppress TensorFlow logging for clarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# The following tf.autograph.experimental.set_verbosity does not exist in recent TF versions and is removed.
# The rest of the code is sufficient for logging suppression.

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)


iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the targets
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training data shape:", X_train.shape, y_train.shape)
print("Test data shape:", X_test.shape, y_test.shape)


# Build a simple dense neural network for tabular data
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

model.save('iris_model.h5')
print("Model saved to iris_model.h5")

# Load the model
loaded_model = keras.models.load_model('iris_model.h5')
print("Model loaded from iris_model.h5")


# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)


# Visualize some predictions
plt.figure(figsize=(10, 4))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.scatter([0], [0], c='white', alpha=0)  # dummy to keep axes
    plt.title(f"Pred: {predicted_classes[i]}\nTrue: {true_classes[i]}")
    plt.axis('off')
plt.suptitle('Iris Test Predictions')
plt.show()

# End of the script
