import numpy as np
import tensorflow as tf


def generate_sin_data(n: int = 10000) -> np.array:
    """Generate data samples from sinus function."""
    x = np.linspace(-np.pi, np.pi, n)
    y = np.sin(x)
    return np.array([[i, j] for i, j in zip(x, y)])


def generate_mnist_data(config):
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32")
    x_train = (x_train - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla-gan"]["batch_size"]
    )

    return train_dataset


def generate_mnist_data_dcgan(config):
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_train = (x_train - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla-gan"]["batch_size"]
    )

    return train_dataset


def load_train_data(config):
    if config["gan-type"] == "dc-gan":
        return generate_mnist_data_dcgan(config)
    elif config["gan-type"] == "vanilla-gan":
        return generate_mnist_data(config)
