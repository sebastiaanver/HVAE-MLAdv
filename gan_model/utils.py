import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt


def display_images(out, label=None, count=False):
    """Inspired by: https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb"""
    out_pic = tf.reshape(out, [-1, 28, 28])
    plt.figure(figsize=(18, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(out_pic[i + 4], cmap="gray")
        plt.axis("off")
        if count:
            plt.title(str(4 + i), color="w")
    plt.show()


def evaluate_gan_is(dataset):
    # Load pre-trained MNIST model
    classifier = tf.keras.Sequential(
        [hub.KerasLayer("https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1")]
    )

    def get_inception_score(classifier, images):
        preds = classifier.predict(images)
        preds = tf.nn.softmax(preds)

        kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        score = np.exp(kl)
        return score

    scores = [get_inception_score(classifier, batch) for batch in dataset]

    return np.mean(scores), np.std(scores)
