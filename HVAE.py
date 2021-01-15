import tensorflow as tf
import math
from typing import List
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
import tensorflow.math as tfm
from utils import log_normal_standard, log_normal_diag


class Encoder(tfkl.Layer):
    def __init__(
        self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        #  TODO: Allow for two different latent dimensions in z1/z2

        self.initializer = tf.keras.initializers.HeNormal()

        # q(z2|x)
        self.dense_proj_z2_x_1 = GatedDense(
            intermediate_dim, name="dense_proj_z2_1"
        )  # Same as q_z2_layers in vampprior
        self.dense_proj_z2_x_2 = GatedDense(
            intermediate_dim, name="dense_proj_z2_2"
        )  # Same as q_z2_layers in vampprior

        self.dense_mean_z2 = tfkl.Dense(latent_dim, kernel_initializer=self.initializer)
        self.dense_log_var_z2 = tfkl.Dense(
            latent_dim, kernel_initializer=self.initializer
        )

        # q(z1|x,z2)
        self.dense_proj_z1_x = GatedDense(intermediate_dim, name="dense_proj_z1_x")
        self.dense_proj_z1_z2 = GatedDense(intermediate_dim, name="dense_proj_z1_z2")

        self.dense_proj_z1_joint = GatedDense(
            intermediate_dim, name="dense_proj_z1_joint"
        )  # Layers that receives both  x and z2.

        self.dense_mean_z1 = tfkl.Dense(latent_dim, kernel_initializer=self.initializer)
        self.dense_log_var_z1 = tfkl.Dense(
            latent_dim, kernel_initializer=self.initializer
        )

        self.sampling = Sampling()

    def call(self, inputs, train=True):
        # z2 ~ q(z2|x)
        x = self.dense_proj_z2_x_1(inputs)
        x = self.dense_proj_z2_x_2(x)

        z2_mean = self.dense_mean_z2(x)
        z2_log_var = self.dense_log_var_z2(x)
        z2_log_var = tf.clip_by_value(
            z2_log_var, clip_value_min=-6.0, clip_value_max=2.0
        )
        z2_return = self.sampling((z2_mean, z2_log_var), train)

        # z1 ~ q(z1|x, z2)
        x = self.dense_proj_z1_x(inputs)
        z2 = self.dense_proj_z1_z2(z2_return)

        joint = self.dense_proj_z1_joint(tf.concat([x, z2], 1))

        z1_mean = self.dense_mean_z1(joint)
        z1_log_var = self.dense_log_var_z1(joint)
        z1_log_var = tf.clip_by_value(
            z1_log_var, clip_value_min=-6.0, clip_value_max=2.0
        )

        z1 = self.sampling((z1_mean, z1_log_var), train)

        return z1, z1_mean, z1_log_var, z2_return, z2_mean, z2_log_var


class Decoder(tfkl.Layer):
    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        input_type_is_binary=True,
        name="decoder",
        **kwargs,
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input_type_is_binary = input_type_is_binary

        self.initializer = tf.keras.initializers.HeNormal()

        # p(z1|z2)
        self.dense_proj_z1_z2_1 = GatedDense(
            intermediate_dim, name="dense_proj_z1_z2_1"
        )
        self.dense_proj_z1_z2_2 = GatedDense(
            intermediate_dim, name="dense_proj_z1_z2_2"
        )

        self.dense_mean_z1 = tfkl.Dense(
            latent_dim,
            activation="sigmoid",
            kernel_initializer=self.initializer,
            name="dense_mean",
        )
        self.dense_logvar_z1 = tfkl.Dense(
            latent_dim, kernel_initializer=self.initializer, name="dense_logvar"
        )

        # p(x|z1, z2)
        self.dense_proj_x_z1 = GatedDense(intermediate_dim, name="dense_proj_x_z1")
        self.dense_proj_x_z2 = GatedDense(intermediate_dim, name="dense_proj_x_z2")
        self.dense_proj_x_joint = GatedDense(
            intermediate_dim, name="dense_proj_x_joint"
        )

        self.dense_output_mean = tfkl.Dense(
            original_dim,
            activation="sigmoid",
            kernel_initializer=self.initializer,
            name="dense_output_mean",
        )
        if not self.input_type_is_binary:
            self.dense_logvar_mean = tfkl.Dense(
                original_dim,
                kernel_initializer=self.initializer,
                name="dense_logvar_mean",
            )

    def call(self, inputs: List):
        z1, z2 = inputs

        # p(z1|z2)
        z2_ = self.dense_proj_z1_z2_1(z2)
        z2_ = self.dense_proj_z1_z2_2(z2_)

        z1_mean = self.dense_mean_z1(z2_)
        z1_log_var = self.dense_logvar_z1(z2_)
        z1_log_var = tf.clip_by_value(
            z1_log_var, clip_value_min=-6.0, clip_value_max=2.0
        )

        # x_mean = p(x|z1,z2)
        z1 = self.dense_proj_x_z1(z1)
        z2 = self.dense_proj_x_z2(z2)

        joint = self.dense_proj_x_joint(tf.concat([z1, z2], 1))

        output_mean = self.dense_output_mean(joint)
        if self.input_type_is_binary:
            output_logvar = 0
        else:
            output_logvar = self.dense_logvar_mean(joint)
            output_logvar = tf.clip_by_value(
                output_logvar, clip_value_min=-4.5, clip_value_max=0.0
            )

        return z1_mean, z1_log_var, output_mean, output_logvar


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
            output_size, use_bias=False,
        )

    def call(self, x):
        x = self.layer(x)
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)

        return x


class HierarchicalVariationalAutoEncoder(tf.keras.Model):
    """From: https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example"""

    def __init__(self, config, nr_inputs=None, name="autoencoder", **kwargs):
        super(HierarchicalVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.config = config["hierarchical"]
        self.dataset = config["dataset_used"]
        self.original_dim = (
            config["dataset"][self.dataset]["rows"]
            * config["dataset"][self.dataset]["cols"]
        )
        print(f"Using {self.config['prior']}")
        self.encoder = Encoder(
            latent_dim=self.config["latent_dim"],
            intermediate_dim=self.config["intermediate_dim"],
        )
        self.decoder = Decoder(
            original_dim=self.original_dim,
            latent_dim=self.config["latent_dim"],
            intermediate_dim=self.config["intermediate_dim"],
            input_type_is_binary=self.config["input_type_is_binary"],
        )
        self.sampling = Sampling()
        self.train = True

        if self.config["prior"] == "vampprior":
            self.nr_inputs = nr_inputs
            self.set_pseudoinputs()

    def eval_model(self):
        self.train = False

    def train_model(self):
        self.train = True

    def sample_and_generate_data(self, n: int):
        if self.config["prior"] == "standard":
            z2_sampled = tf.random.normal(
                [n, self.config["latent_dim"]], 0, 1, tf.float32, seed=42
            )
        elif self.config["prior"] == "vampprior":
            print(self.idle_inputs[0:1])
            print(self.means(self.idle_inputs[0:1]))
            print(self.idle_inputs[1:2])
            print(self.means(self.idle_inputs[1:2]))
            means = self.means(self.idle_inputs)[0:n]
            _, _, _, z2_sampled, _, _ = self.encoder(means)

        # p(z1|z2)
        z2 = self.decoder.dense_proj_z1_z2_1(z2_sampled)
        z2 = self.decoder.dense_proj_z1_z2_2(z2)

        z1_mean = self.decoder.dense_mean_z1(z2)
        z1_log_var = self.decoder.dense_logvar_z1(z2)
        z1 = self.sampling((z1_mean, z1_log_var), False)

        # x_mean = p(x|z1,z2)
        z1 = self.decoder.dense_proj_x_z1(z1)
        z2 = self.decoder.dense_proj_x_z2(z2_sampled)

        joint = self.decoder.dense_proj_x_joint(tf.concat([z1, z2], 1))

        output_mean = self.decoder.dense_output_mean(joint)

        return output_mean

    def set_pseudoinputs(self):
        """
        TODO: Check Hardtanh, which is used in the paper in the layer
        """
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
            tf.eye(
                self.config["nr_components"] if not self.nr_inputs else self.nr_inputs
            ),
            trainable=False,
        )  # This is used to select a number of pseudo inputs from self.means.

    def get_number_of_active_units(self, x):
        _, z1_mean, _, _, z2_mean, _ = self.encoder(x)
        variances_z1 = tfp.stats.variance(z1_mean, sample_axis=0).numpy()
        active_units_z1 = [variance for variance in variances_z1 if variance > 1e-2]

        variances_z2 = tfp.stats.variance(z2_mean, sample_axis=0).numpy()
        active_units_z2 = [variance for variance in variances_z2 if variance > 1e-2]

        print(
            f"There are {len(active_units_z1)} active units out of the {len(variances_z1)} units  in z1."
        )
        print(
            f"There are {len(active_units_z2)} active units out of the {len(variances_z2)} units in z2."
        )

        return len(active_units_z1), len(active_units_z2)

    def get_prior(self, z):
        if self.config["prior"] == "standard":
            log_prior = log_normal_standard(z, axis=1)

        elif self.config["prior"] == "vampprior":
            # z - MB x M
            C = self.config["nr_components"] if not self.nr_inputs else self.nr_inputs

            # calculate params
            X = self.means(self.idle_inputs)

            # calculate params for given data
            _, _, _, _, z2_mean, z2_log_var = self.encoder(X)  # C x M

            # expand z
            z_expand = tf.expand_dims(z, axis=1)
            means = tf.expand_dims(z2_mean, axis=0)
            logvars = tf.expand_dims(z2_log_var, axis=0)

            a = log_normal_diag(z_expand, means, logvars, axis=2) - math.log(
                C
            )  # MB x C
            a_max = tf.math.reduce_max(a, axis=1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + tfm.log(
                tfm.reduce_sum(tfm.exp(a - tf.expand_dims(a_max, 1)), 1)
            )  # MB x 1

        return log_prior

    def call(self, inputs, training=False):
        (
            z1_enc,
            z1_mean_enc,
            z1_log_var_enc,
            z2_enc,
            z2_mean_enc,
            z2_log_var_enc,
        ) = self.encoder(inputs, self.train)

        z1_mean_dec, z1_log_var_dec, output_mean_dec, output_logvar_dec = self.decoder(
            [z1_enc, z2_enc]
        )

        # KL
        log_p_z1 = log_normal_diag(z1_enc, z1_mean_dec, z1_log_var_dec, axis=1)
        log_q_z1 = log_normal_diag(z1_enc, z1_mean_enc, z1_log_var_enc, axis=1)
        log_p_z2 = self.get_prior(z2_enc)
        log_q_z2 = log_normal_diag(z2_enc, z2_mean_enc, z2_log_var_enc, axis=1)

        return (
            output_mean_dec,
            output_logvar_dec,
            log_p_z1,
            log_q_z1,
            log_p_z2,
            log_q_z2,
        )
