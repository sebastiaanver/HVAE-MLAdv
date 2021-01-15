import tensorflow as tf
import yaml
from pathlib import Path

from utils import (
    display_images,
    loss_function,
    load_frey_faces_data,
    load_fashion_mnist_data,
)
from VAE_vanilla import VariationalAutoEncoder

config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

vae = VariationalAutoEncoder(config)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

loss_metric = tf.keras.metrics.Mean()
train_dataset, test_dataset = load_frey_faces_data(config)

for epoch in range(config["vanilla"]["epochs"]):
    print(f"Epoch nr: {epoch}")

    print("Training...")
    vae.train_model()
    for step, train_batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:

            reconstructed_mean, reconstructed_var, z, log_p_z, log_q_z = vae(
                train_batch
            )

            # Compute loss
            loss = loss_function(
                reconstructed_mean,
                reconstructed_var,
                train_batch,
                log_p_z,
                log_q_z,
                binary=config["vanilla"]["input_type_is_binary"],
            )

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print(f"step {step}: mean loss = {loss_metric.result()}")

    display_images(train_batch, reconstructed_mean, config)

    vae.eval_model()
    print("Testing...")
    for step, test_batch in enumerate(test_dataset):
        reconstructed_mean, reconstructed_var, z, log_p_z, log_q_z = vae(test_batch)

        # Compute loss
        loss = loss_function(
            reconstructed_mean,
            reconstructed_var,
            test_batch,
            log_p_z,
            log_q_z,
            binary=config["vanilla"]["input_type_is_binary"],
        )

        loss_metric(loss)
        if step % 100 == 0:
            print(f"step {step}: mean loss = {loss_metric.result()}")

    display_images(test_batch, reconstructed_mean, config)
