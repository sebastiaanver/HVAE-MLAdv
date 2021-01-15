import tensorflow as tf
import tensorflow.math as tfm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import tensorflow_hub as hub
import matplotlib.gridspec as gridspec
import urllib
from scipy.io import loadmat
import numpy as np
from scipy.special import logsumexp
import os


def log_normal_diag(x, mean, log_var, average=False, axis=None) -> Any:
    log_normal = -0.5 * (log_var + tfm.pow(x - mean, 2) / tfm.exp(log_var))
    if average:
        return tfm.reduce_mean(log_normal, axis=axis)
    else:
        return tfm.reduce_sum(log_normal, axis=axis)


def log_normal_standard(x, average=False, axis=None) -> Any:
    log_normal = -0.5 * tfm.pow(x, 2)
    if average:
        return tfm.reduce_mean(log_normal, axis=axis)
    else:
        return tfm.reduce_sum(log_normal, axis=axis)


def log_logistic_256(x, mean, logvar, average=False, reduce=True, axis=None) -> Any:
    bin_size = 1.0 / 256.0
    scale = tf.exp(logvar)
    sample = (tf.floor(x / bin_size) * bin_size - mean) / scale
    logp = -tfm.log(tf.sigmoid(sample + bin_size / scale) - tfm.sigmoid(sample) + 1e-7)

    if reduce:
        if average:
            return tfm.reduce_mean(logp, axis=axis)
        else:
            return tfm.reduce_sum(logp, axis=axis)
    else:
        return logp


def log_bernoulli(x, mean, average=False, axis=None) -> Any:
    probs = tf.clip_by_value(mean, clip_value_min=1e-5, clip_value_max=1.0 - 1e-5)
    log_bernoulli = x * tfm.log(probs) + (1.0 - x) * tfm.log(1.0 - probs)
    if average:
        return tfm.reduce_mean(log_bernoulli, axis)
    else:
        return tfm.reduce_sum(log_bernoulli, axis)


def display_images(in_, out, config: Dict, label=None, count=False) -> None:
    """Inspired by: https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb"""
    dataset = config["dataset_used"]
    img_rows, img_cols = (
        config["dataset"][dataset]["rows"],
        config["dataset"][dataset]["cols"],
    )
    if in_ is not None:
        in_pic = tf.reshape(in_, [-1, img_rows, img_cols])
        plt.figure(figsize=(18, 4))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(in_pic[i + 4], cmap='gray')
            plt.axis("off")
    out_pic = tf.reshape(out, [-1, img_rows, img_cols])
    plt.figure(figsize=(18, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(out_pic[i + 4], cmap='gray')
        plt.axis("off")
        if count:
            plt.title(str(4 + i), color="w")
    plt.show()


def display_images_grid(out, config: Dict, save=False) -> None:
    dataset = config["dataset_used"]
    plt.figure(figsize=(10,10))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.

    img_rows, img_cols = (
        config["dataset"][dataset]["rows"],
        config["dataset"][dataset]["cols"],
    )
    out_pic = tf.reshape(out, [-1, img_rows, img_cols])

    for i in range(16):
        plt.subplot(gs1[i])
        plt.imshow(out_pic[i], cmap='gray')
        plt.axis("off")
    if save:
        plt.savefig("img.jpg")
    else:
        plt.show()


def loss_function(x_mean, x_logvar, x, log_p_z, log_q_z, binary=True) -> Any:
    kl_loss = -(log_p_z - log_q_z)

    if binary:
        reconstruction_error = log_bernoulli(x, x_mean, axis=1)

    else:
        reconstruction_error = -log_logistic_256(x, x_mean, x_logvar, axis=1)

    return kl_loss - reconstruction_error


def loss_function_hvae(
    x, x_mean, x_logvar, log_p_z1, log_q_z1, log_p_z2, log_q_z2, binary=True
) -> Any:
    kl_loss = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

    if binary:
        reconstruction_error = log_bernoulli(x, x_mean, axis=1)

    else:
        reconstruction_error = -log_logistic_256(x, x_mean, x_logvar, axis=1)

    return kl_loss - reconstruction_error


def calculate_likelihood(
    vae, test_dataset, model_type: str, config: Dict, S=5000, MB=100
):
    """
    Args:
        vae: Trained model.
        test_dataset:  Data.
        S: Number of samples used for approximating log-likelihood.
        MB: Size of a mini-batch used for approximating log-likelihood.

    """
    for test_batch in test_dataset:
        data = test_batch
        N_test = data.shape[0]
        likelihood_test = []

        #  If the number of samples is lower than the sample in batch
        if S <= MB:
            R = 1

        else:
            R = S / MB  # Number of runs
            S = MB  # Number of samples per run

        # Loop over all items in test set X
        for j in range(N_test):
            if j % 100 == 0:
                print("{:.2f}%".format(j / (1.0 * N_test) * 100))

            # Take x*
            x_single = tf.expand_dims(
                data[j], 0
            )

            a = []
            for r in range(int(R)):
                # Repeat it for all training points
                x = tf.repeat(
                    x_single, [S], axis=0
                )  # Same as expand in Pytorch (just multiplying it)

                if model_type == "hierarchical":
                    (
                        output_mean_dec,
                        output_logvar_dec,
                        log_p_z1,
                        log_q_z1,
                        log_p_z2,
                        log_q_z2,
                    ) = vae(x)
                    a_tmp = loss_function_hvae(
                        x,
                        output_mean_dec,
                        output_logvar_dec,
                        log_p_z1,
                        log_q_z1,
                        log_p_z2,
                        log_q_z2,
                        binary=config["hierarchical"]["input_type_is_binary"],
                    )
                else:
                    reconstructed_mean, reconstructed_var, z, log_p_z, log_q_z = vae(x)
                    a_tmp = loss_function(
                        reconstructed_mean,
                        reconstructed_var,
                        x,
                        log_p_z,
                        log_q_z,
                        binary=config["vanilla"]["input_type_is_binary"],
                    )

                a.append(-a_tmp.numpy())

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp(a)
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

    return -np.mean(likelihood_test)


def load_mnist_data(config: Dict) -> Tuple[Any]:
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    img_rows, img_cols = (
        config["dataset"]["mnist"]["rows"],
        config["dataset"]["mnist"]["cols"],
    )

    x_train = x_train.reshape(-1, img_rows * img_cols).astype("float32") / 255
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla"]["batch_size"]
    )

    x_test = x_test.reshape(-1, img_rows * img_cols).astype("float32") / 255
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(x_test.shape[0])

    return train_dataset, test_dataset


def load_fashion_mnist_data(config: Dict) -> Tuple[Any]:
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    img_rows, img_cols = (
        config["dataset"]["fashion_mnist"]["rows"],
        config["dataset"]["fashion_mnist"]["cols"],
    )

    x_train = x_train.reshape(-1, img_rows * img_cols).astype("float32") / 255
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla"]["batch_size"]
    )

    x_test = x_test.reshape(-1, img_rows * img_cols).astype("float32") / 255
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(x_test.shape[0])

    return train_dataset, test_dataset


def load_frey_faces_data(config: Dict) -> Tuple[Any]:
    url = "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    img_rows, img_cols = (
        config["dataset"]["frey_faces"]["rows"],
        config["dataset"]["frey_faces"]["cols"],
    )
    filename = os.path.join("data/Freyfaces/", os.path.basename(url))

    if not os.path.exists(filename):
        f = urllib.request.urlopen(url)

        with open(filename, "wb") as local_file:
            local_file.write(f.read())

    frey_faces = loadmat(filename, squeeze_me=True, struct_as_record=False)
    frey_faces = frey_faces["ff"].T.reshape((-1, img_rows * img_cols)) / 255

    train_test_split = int(frey_faces.shape[0] * 0.7)
    x_train, x_test = frey_faces[:train_test_split], frey_faces[train_test_split + 1 :]

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla"]["batch_size"]
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(x_test.shape[0])

    return train_dataset, test_dataset


def load_omniglot_data(config: Dict) -> Tuple[Any]:
    img_rows, img_cols = (
        config["dataset"]["omniglot"]["rows"],
        config["dataset"]["omniglot"]["cols"],
    )
    omni_raw = loadmat(
        os.path.join("data", "OMNIGLOT", "chardata.mat"),
        squeeze_me=True,
        struct_as_record=False,
    )

    train_data, test_data = omni_raw["data"], omni_raw["testdata"]
    train_data = train_data.T.reshape((-1, img_rows * img_cols))
    test_data = test_data.T.reshape((-1, img_rows * img_cols))

    x_train = tf.convert_to_tensor(train_data, dtype=tf.float32)
    x_test = tf.convert_to_tensor(test_data, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        config["vanilla"]["batch_size"]
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(x_test.shape[0])

    return train_dataset, test_dataset


def load_data(config: Dict) -> Tuple[Any]:
    if config["dataset_used"] == "fashion_mnist":
        return load_fashion_mnist_data(config)
    elif config["dataset_used"] == "mnist":
        return load_mnist_data(config)
    elif config["dataset_used"] == "frey_faces":
        return load_frey_faces_data(config)
    elif config["dataset_used"] == "omniglot":
        return load_omniglot_data(config)


def evaluate_is(dataset):
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
