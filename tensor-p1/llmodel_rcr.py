import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import os
import PyPDF2

# Extract text from all PDFs in rcr_docs folder
pdf_folder = os.path.join(os.path.dirname(__file__), 'rcr_docs')
pdf_text = ''
for fname in os.listdir(pdf_folder):
    if fname.lower().endswith('.pdf'):
        with open(os.path.join(pdf_folder, fname), 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pdf_text += page.extract_text() or ''

# Print a sample of the extracted text
print('Sample extracted text:')
print(pdf_text[:1000])



text = pdf_text
if len(text) < 10:
    raise ValueError("Extracted text is too short for training. Check your PDF content.")
chars = sorted(list(set(text)))
print(f"Unique characters in text: {chars}")
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}


# Prepare input/output pairs
seq_length = 10
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

# Temperature sampling for more creative output
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, length=200, temperature=1.0):
    generated = seed
    for _ in range(length):
        x_pred = np.array([[char_indices.get(c, 0) for c in generated[-seq_length:]]])
        x_pred = x_pred.reshape((1, seq_length, 1)) / float(len(chars))
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated += next_char
    return generated


# Prompt-based generation
while True:
    prompt = input(f"Enter a prompt of at least {seq_length} characters (or 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    if len(prompt) < seq_length:
        print(f"Prompt too short. Please enter at least {seq_length} characters.")
        continue
    temp = input("Temperature (default 1.0, lower=more predictable, higher=more random): ")
    try:
        temp = float(temp)
    except Exception:
        temp = 1.0
    print("\nGenerated text:\n")
    print(generate_text(prompt[:seq_length], length=300, temperature=temp))
    print("\n---\n")
