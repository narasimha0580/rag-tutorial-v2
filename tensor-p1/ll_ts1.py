import tensorflow as tf
import datetime

print('Step 1: Importing libraries')

print('Step 2: Loading MNIST dataset')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
print('First training label:', y_train[0])
print('First training image (as array):\n', x_train[0])

print('Step 3: Normalizing pixel values to [0, 1] range')
x_train, x_test = x_train / 255.0, x_test / 255.0
print('First training image after normalization:\n', x_train[0])

def create_model():
    print('Step 4: Building the model')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print('Model summary:')
    model.summary()
    return model

model = create_model()

print('Step 5: Compiling the model (setting up how it learns)')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('Step 6: Starting training...')
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

print('Step 7: Training complete!')
print('Final training accuracy:', history.history['accuracy'][-1])
print('Final validation accuracy:', history.history['val_accuracy'][-1])
print('Final training loss:', history.history['loss'][-1])
print('Final validation loss:', history.history['val_loss'][-1])

print('Step 8: Evaluating the model on test data...')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

print('Step 9: Predicting the first test image...')
predictions = model.predict(x_test)
print('Model raw output (probabilities) for first test image:', predictions[0])
print("Model's predicted digit:", tf.argmax(predictions[0]).numpy())
print('Actual digit:', y_test[0])

