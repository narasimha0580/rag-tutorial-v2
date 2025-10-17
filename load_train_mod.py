import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random 

# 1️⃣ Load MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2️⃣ Normalize like in training
x_test = x_test / 255.0

# 3️⃣ Load the trained model
model = load_model("mnist_model.h5")

# # 4️⃣ Pick one sample (e.g., first image)
# index = 0
# sample = x_test[index].reshape(1, 28, 28)

# # 5️⃣ Predict
# prediction = np.argmax(model.predict(sample))
# print(f"Predicted digit: {prediction}")
# print(f"Actual digit: {y_test[index]}")

# # 6️⃣ Optional: Show the image
# plt.imshow(x_test[index], cmap='gray')
# plt.title(f"Predicted: {prediction} | Actual: {y_test[index]}")
# plt.axis('off')
# plt.show()


# def eval_dig(index):
#     sample = x_test[index].reshape(1, 28, 28)

#     # 5️⃣ Predict
#     prediction = np.argmax(model.predict(sample))
#     print(f"Predicted digit: {prediction}")
#     print(f"Actual digit: {y_test[index]}")

#     # 6️⃣ Optional: Show the image
#     plt.imshow(x_test[index], cmap='gray')
#     plt.title(f"Predicted: {prediction} | Actual: {y_test[index]}")
#     plt.axis('off')
#     plt.show()
#     time.sleep(1)
#     plt.close()


def eval_dig(index, delay=1.0):
    sample = x_test[index].reshape(1, 28, 28)
    prediction = np.argmax(model.predict(sample))
    # print(f"Predicted digit: {prediction}")
    # print(f"Actual digit: {y_test[index]}")


    if not prediction == y_test[index]:
        print(f'{prediction} != {y_test[index]}')

    plt.figure(figsize=(3,3))
    
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Predicted: {prediction} | Actual: {y_test[index]}")
    plt.axis('off')
    plt.draw()          # Draw the plot
    plt.pause(delay)    # Display for 'delay' seconds
    plt.close()         # Then close automatically



for i in range(100):
    eval_dig(random.randrange(1,10000), delay=2)
