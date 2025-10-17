import tensorflow as tf
from tensorflow.keras import layers, models

# 1️⃣ Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Normalize pixel values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3️⃣ Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),     # Flatten 28x28 images
    layers.Dense(128, activation='relu'),     # Hidden layer
    layers.Dropout(0.2),                      # Regularization
    layers.Dense(10, activation='softmax')    # Output layer (10 digits)
])

# 4️⃣ Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5️⃣ Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 6️⃣ Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# 7️⃣ Save the model
model.save("mnist_model.h5")
print("💾 Model saved as mnist_model.h5")
