import tensorflow as tf
from typing import Dict
from pathlib import Path
import yaml
from VAE_vanilla import VariationalAutoEncoder
from utils import loss_function, display_images


def main(config: Dict) -> None:
    original_dim = 28 * 28
    batch_size = config["vanilla"]["batch_size"]
    prior = config["vanilla"]["prior"]
    vae = VariationalAutoEncoder(original_dim, 400, 20, prior)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss_metric = tf.keras.metrics.Mean()

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=1024)

    epochs = config["vanilla"]["epochs"]

    for epoch in range(epochs):
        print(f"Epoch nr: {epoch}")

        print("Training...")
        vae.train_model()
        for step, train_batch in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                reconstructed, log_p_z, log_q_z = vae(train_batch)

                # Compute reconstruction loss
                loss = loss_function(train_batch, reconstructed, log_p_z, log_q_z)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print(f"step {step}: mean loss = {loss_metric.result()}")

        display_images(test_batch, reconstructed)
        vae.eval_model()
        print("Testing...")
        for step, test_batch in enumerate(test_dataset):
            reconstructed = vae(test_batch)

            # Compute reconstruction loss
            loss = loss_function(test_batch, reconstructed, log_p_z, log_q_z)

            loss_metric(loss)
            if step % 100 == 0:
                print(f"step {step}: mean loss = {loss_metric.result()}")
        display_images(test_batch, reconstructed)


if __name__ == "__main__":
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
    main(config)
