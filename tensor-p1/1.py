import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Flatten, Conv2D
from keras.datasets import mnist
from keras import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import time
import logging
import sys
import datetime
import json
import random
import math
import re



# Suppress TensorFlow logging for clarity

# Suppress TensorFlow logging for clarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# The following tf.autograph.experimental.set_verbosity does not exist in recent TF versions and is removed.
# The rest of the code is sufficient for logging suppression.

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)




# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() 

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2) 

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
# Save the model
model.save('mnist_cnn_model.h5')
print("Model saved to mnist_cnn_model.h5")  

# Load the model
loaded_model = keras.models.load_model('mnist_cnn_model.h5')
print("Model loaded from mnist_cnn_model.h5")   

# Make predictions with the loaded model
predictions = loaded_model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print("Loaded model accuracy:", accuracy)   

# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}, True: {true_classes[i]}")
    plt.axis('off')
plt.show()  

# Clean up saved model file
os.remove('mnist_cnn_model.h5')
print("Cleaned up saved model file.")   

# End of the script
