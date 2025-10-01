import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Sample text data (tiny corpus)
text = "hello world"
chars = sorted(list(set(text)))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}

# Prepare input/output pairs
seq_length = 3
X = []
y = []
for i in range(len(text) - seq_length):
    seq = text[i:i+seq_length]
    target = text[i+seq_length]
    X.append([char_indices[c] for c in seq])
    y.append(char_indices[target])
X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

# Reshape X for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], seq_length, 1))
X = X / float(len(chars))  # normalize

# Build model
model = Sequential([
    LSTM(32, input_shape=(seq_length, 1)),
    Dense(len(chars), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Generate text
def generate_text(seed, length=20):
    generated = seed
    for _ in range(length):
        x_pred = np.array([[char_indices[c] for c in generated[-seq_length:]]])
        x_pred = x_pred.reshape((1, seq_length, 1)) / float(len(chars))
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        generated += next_char
    return generated

# Example usage
seed = "hello"
print("Seed:", seed)
print("Generated:", generate_text(seed, length=20))
