from tensorflow import keras
from matplotlib import pyplot as plt

def load_mnist_data():
    """Load and preprocess the MNIST dataset.

    Returns:
        Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dimensions to include channel information
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return (x_train, y_train), (x_test, y_test)


def disp_plot():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # disp_plot()


    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    print(f"Number of training samples: {x_train.shape[0]}")
    print(f"Number of test samples: {x_test.shape[0]}") 

