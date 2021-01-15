import tensorflow as tf
import yaml
from pathlib import Path
from models import GAN
from utils import display_images, evaluate_gan_is
from data_gen import load_train_data


def train(model, config):
    epochs = config["epochs"]
    train_dataset = load_train_data(config)

    for epoch in range(epochs):
        print(f"Epoch nr: {epoch}")
        for step, train_batch in enumerate(train_dataset):
            discriminator_loss, generator_loss = model(train_batch)

            if step % 100 == 0:
                print(
                    f"Step {step}: Discriminator loss = {discriminator_loss}, Generator loss: {generator_loss}"
                )

        generated_samples = model.generate_sample()
        display_images(generated_samples)


def evaluate(model):
    dataset = [model.generate_sample() for _ in range(100)]
    dataset = tf.reshape(dataset, (-1, 28, 28, 1))
    dataset = (
        tf.data.Dataset.from_tensor_slices(dataset).shuffle(buffer_size=1024).batch(100)
    )
    mean, var = evaluate_gan_is(dataset)

    print(f"Inception  score mean: {mean}, with variance {var}.")


def main(config):
    model = GAN(config)
    train(model, config)
    model.save(f"saved_models/gan_{config['gan-type']}")
    evaluate(model)


if __name__ == "__main__":
    model_config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
    main(model_config)
