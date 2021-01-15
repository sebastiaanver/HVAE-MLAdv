import tensorflow as tf
import math
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
import tensorflow.math as tfm
from utils import log_normal_standard, log_normal_diag


class Encoder(tfkl.Layer):
    def __init__(
        self,
        latent_dim=32,
        intermediate_dim=64,
        prior="standard",
        name="encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        # self.dense_proj = tfkl.Dense(intermediate_dim, activation="relu")

        self.initializer = tf.keras.initializers.HeNormal()
        self.dense_proj_1 = GatedDense(
            intermediate_dim, name="dense_proj_1"
        )  # Same as q_z_layers in vampprior
        self.dense_proj_2 = GatedDense(
            intermediate_dim, name="dense_proj_2"
        )  # Same as q_z_layers in vampprior

        self.dropout_layer = tfkl.Dropout(rate=0.2)

        self.dense_mean = tfkl.Dense(latent_dim, kernel_initializer=self.initializer)
        self.dense_log_var = tfkl.Dense(latent_dim, kernel_initializer=self.initializer)

        self.sampling = Sampling()
        self.prior = prior

    def call(self, inputs, training=False):
        x = self.dense_proj_1(inputs)
        # if training:
        #     x = self.dropout_layer(x)
        x = self.dense_proj_2(x)
        # if training:
        #     x = self.dropout_layer(x)
        z_mean = self.dense_mean(x)
        # if training:
        #     z_mean = self.dropout_layer(z_mean)
        z_log_var = self.dense_log_var(x)
        z_log_var = tf.clip_by_value(z_log_var, clip_value_min=-6.0, clip_value_max=2.0)
        # if training:
        #     z_log_var = self.dropout_layer(z_log_var)
        z = self.sampling((z_mean, z_log_var), training)
        return z_mean, z_log_var, z


class Decoder(tfkl.Layer):
    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        input_type_is_binary=True,
        name="decoder",
        **kwargs,
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input_type_is_binary = input_type_is_binary

        self.initializer = tf.keras.initializers.HeNormal()

        self.dense_proj_1 = GatedDense(
            intermediate_dim, name="dense_proj_1"
        )  # Same as q_z_layers in vampprior
        self.dense_proj_2 = GatedDense(
            intermediate_dim, name="dense_proj_1"
        )  # Same as q_z_layers in vampprior

        self.dense_output_mean = tfkl.Dense(
            original_dim,
            activation="sigmoid",
            kernel_initializer=self.initializer,
            name="dense_mean",
        )
        if not self.input_type_is_binary:
            self.dense_output_logvar = tfkl.Dense(
                original_dim, kernel_initializer=self.initializer, name="dense_logvar"
            )

    def call(self, inputs):
        x = self.dense_proj_1(inputs)
        x = self.dense_proj_2(x)
        output_mean = self.dense_output_mean(x)
        if self.input_type_is_binary:
            output_logvar = 0
        else:
            output_mean = tf.clip_by_value(
                output_mean,
                clip_value_min=1.0 / 512.0,
                clip_value_max=1.0 - 1.0 / 512.0,
            )
            output_logvar = self.dense_output_logvar(x)
            output_logvar = tf.clip_by_value(
                output_logvar, clip_value_min=-4.5, clip_value_max=0.0
            )
        return output_mean, output_logvar


class Sampling(tfkl.Layer):
    def call(self, inputs, train=True):
        z_mean, z_log_var = inputs
        if train:
            epsilon = tf.keras.backend.random_normal(shape=z_log_var.shape)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        else:
            return z_mean


class GatedDense(tfkl.Layer):
    def __init__(self, output_size, name, **kwargs):
        super(GatedDense, self).__init__(name=name, **kwargs)

        self.initializer = tf.keras.initializers.HeNormal()
        self.h = tfkl.Dense(
            output_size, name=name + "gated_h", kernel_initializer=self.initializer
        )
        self.g = tfkl.Dense(
            output_size,
            activation="sigmoid",
            kernel_initializer=self.initializer,
            name=name + "gated_g",
        )

    def call(self, x):
        h = self.h(x)
        g = self.g(x)

        return h * g


class PseudoInputs(tfkl.Layer):
    def __init__(self, output_size, initializer, name, **kwargs):
        super(PseudoInputs, self).__init__(name=name, **kwargs)

        self.layer = tfkl.Dense(
            output_size, use_bias=False, input_shape=(500, 500)
        )

    def call(self, x):
        x = self.layer(x)
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)

        return x


class VariationalAutoEncoder(tf.keras.Model):
    """From: https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example"""

    def __init__(self, config, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.config = config["vanilla"]
        self.dataset = config["dataset_used"]
        self.original_dim = (
            config["dataset"][self.dataset]["rows"]
            * config["dataset"][self.dataset]["cols"]
        )
        self.encoder = Encoder(
            latent_dim=self.config["latent_dim"],
            intermediate_dim=self.config["intermediate_dim"],
            prior=self.config["prior"],
        )
        self.decoder = Decoder(
            original_dim=self.original_dim,
            intermediate_dim=self.config["intermediate_dim"],
            input_type_is_binary=self.config["input_type_is_binary"],
        )
        self.train = True

        if self.config["prior"] == "vampprior":
            self.set_pseudoinputs()

    def eval_model(self):
        self.train = False

    def train_model(self):
        self.train = True

    def sample_and_generate_data(self, n: int):
        if self.config["prior"] == "standard":
            z = tf.random.normal(
                [n, self.config["latent_dim"]], 0, 1, tf.float32, seed=42
            )
        elif self.config["prior"] == "vampprior":
            means = self.means(self.idle_inputs)[0:n]
            _, _, z = self.encoder(means)

        output_mean, _ = self.decoder(z)
        return output_mean

    def set_pseudoinputs(self):
        # Initialize layer weights.
        if self.config["use_training_data_init"]:
            initializer = tf.keras.initializers.Constant(
                value=self.config["pseudoinputs_mean"]
            )
        else:
            initializer = tf.keras.initializers.RandomNormal(
                mean=self.config["pseudoinputs_mean"],
                stddev=self.config["pseudoinputs_stddev"],
            )
        self.means = PseudoInputs(self.original_dim, initializer, name="pseudo_inputs")

        self.idle_inputs = tf.Variable(
            tf.eye(self.config["nr_components"]), trainable=False
        )  # This is used to select a number of pseudo inputs from self.means.

    def get_number_of_active_units(self, x):
        z_mean, _, _ = self.encoder(x)
        variances = tfp.stats.variance(z_mean, sample_axis=0).numpy()
        active_units = [variance for variance in variances if variance > 1e-2]

        print(
            f"There are {len(active_units)} active units out of the {len(variances)} units."
        )

        return len(active_units)

    def get_prior(self, z):
        if self.config["prior"] == "standard":
            log_prior = log_normal_standard(z, axis=1)

        elif self.config["prior"] == "vampprior":
            # calculate params
            X = self.means(self.idle_inputs)

            # calculate params for given data
            z_p_mean, z_p_logvar, _ = self.encoder(X)  # C x M

            # expand z
            z_expand = tf.expand_dims(z, 1)
            means = tf.expand_dims(z_p_mean, 0)
            logvars = tf.expand_dims(z_p_logvar, 0)

            a = log_normal_diag(z_expand, means, logvars, axis=2) - math.log(
                self.config["nr_components"]
            )  # MB x C
            a_max = tf.math.reduce_max(a, axis=1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + tfm.log(
                tfm.reduce_sum(tfm.exp(a - tf.expand_dims(a_max, 1)), 1)
            )  # MB x 1

        return log_prior

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training)
        reconstructed_mean, reconstructed_var = self.decoder(z)

        # KL as implemented in the vampprior paper
        log_p_z = self.get_prior(z)  # The prior
        log_q_z = log_normal_diag(
            z, z_mean, z_log_var, axis=1
        )  # Distribution from Encoder.

        return reconstructed_mean, reconstructed_var, z, log_p_z, log_q_z
