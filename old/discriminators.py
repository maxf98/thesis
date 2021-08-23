from abc import ABC, abstractmethod
import math
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
import tensorflow_probability as tfp


class SkillDiscriminator(ABC):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim):
        """performs skill discovery, i.e. mapping inputs to latents,
         either via unsupervised (VAE) or supervised (MLP) training"""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def train(self, x, z=None, batch_size=32, epochs=1):
        """perform discriminator(/generator) training"""

    @abstractmethod
    def call(self, x):
        """return encoded (sampled) values"""

    @abstractmethod
    def log_probs(self, x, z):
        """evaluate likelihood of mapping x to z under discriminator"""

    @abstractmethod
    def save(self, log_dir):
        """saves the underlying discriminator model (not necessarily keras model, not necessarily need to be saved)"""


class UniformCategoricalDiscriminator(SkillDiscriminator):
    def __init__(self,
                 input_dim,
                 hidden_dim=(128, 128),
                 latent_dim=4):
        """
        maps inputs to a uniform categorical distribution, i.e. one_hot vector with latent_dim categories
        performs supervised training using ground truth labels
        """
        super(UniformCategoricalDiscriminator, self).__init__(input_dim, hidden_dim, latent_dim)

        inputs = tfkl.Input(shape=(input_dim, ))
        x = inputs
        for hd in hidden_dim:
            x = tfkl.Dense(hd, activation='relu')(x)
        z_enc = tfkl.Dense(latent_dim, activation='softmax')(x)

        self.model = tfk.Model(inputs, z_enc, name='discriminator')

        self.model.compile(
            loss=tfk.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def train(self, x, z=None, batch_size=32, epochs=1):
        return self.model.fit(x, z, batch_size=batch_size, epochs=epochs)

    def call(self, x, return_probs=False):
        probs = self.model.predict(x)
        if return_probs:
            return probs
        return tf.one_hot([tf.argmax(probs, axis=-1)], self.latent_dim)

    def log_probs(self, x, z, return_full=False):
        """
        expects z as one-hot vector
        """
        probs = self.model.predict(x)

        if not return_full:
            probs = tf.reduce_max(tf.multiply(probs, z), axis=-1)

        return tf.math.log(probs)

    def save(self, log_dir):
        self.model.save(log_dir)


class GaussianDiscriminator(SkillDiscriminator):
    def __init__(self,
                 input_dim,
                 hidden_dim=(128, 128),
                 latent_dim=4,
                 batch_norm=False,
                 fix_variance=False
                 ):
        super(GaussianDiscriminator, self).__init__(input_dim, hidden_dim, latent_dim)

        inputs = tfkl.Input(shape=(input_dim,))
        x = inputs
        for hd in hidden_dim:
            x = tfkl.Dense(hd, activation='relu')(x)
            if batch_norm:
                x = tfkl.BatchNormalization()(x)

        params_size = tfpl.MixtureSameFamily.params_size(self.latent_dim,
                                                         component_params_size=tfpl.IndependentNormal.params_size([1]))
        x = tfkl.Dense(params_size, activation=None)(x)
        outputs = tfpl.MixtureSameFamily(self.latent_dim, tfpl.IndependentNormal([1]))(x)

        self.model = tfk.Model(inputs, outputs, name='discriminator')

        self.model.compile(
            loss=lambda y, model: -model.log_prob(y),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def train(self, x, z=None, batch_size=32, epochs=1):
        return self.model.fit(x, z, batch_size=batch_size, epochs=epochs)

    def call(self, x):
        # I think this automatically samples from the distribution produced in the output layer,
        # rather than returning the distribution
        pred = self.model.predict(x)
        return pred

    def log_probs(self, x, z):
        distr = self.model(x)
        return distr.log_prob(z)

    def save(self, log_dir):
        self.model.save(log_dir)


class OracleDiscriminator:
    def __init__(self, input_shape, intermediate_dim, num_skills, load_from=None):
        # params immediately discarded, only to make interchangeable with DIAYNDiscrim
        self.num_skills = num_skills  # for now we assume it's 4

    def train(self, x, y):
        pass

    def _prob(self, point):
        x, y = point
        msp = 0.001

        if x >= 0:
            if y >= 0:
                ret = [y / (x + y), x / (x + y), msp, msp]
            else:
                ret = [msp, x / (x - y), -y / (x - y), msp]
        else:
            if y >= 0:
                ret = [y / (y - x), msp, msp, -x / (y - x)]
            else:
                ret = [msp, msp, y / (x + y), x / (x + y)]

        if math.fabs(x) < msp and math.fabs(y) < msp:
            ret = [0.25, 0.25, 0.25, 0.25]

        dist_scaling_factor = 1 + (x**2 + y **2)

        return ret * dist_scaling_factor

    def _simple_prob(self, point):
        x, y = point
        msp = 0.001
        return (1 + 10 *(x**2 + y **2)) * np.array([1-msp if x>=0 and y>=0 else msp,
                         1-msp if x>=0 and y<0 else msp,
                         1-msp if x<0 and y<0 else msp,
                         1-msp if x<0 and y>=0 else msp])


    def call(self, batch):
        """
        Parameters
        ----------
        batch – batch expected as batch_size * 2

        Returns
        prob of each skill for each data_point in batch_size * 4
        """
        probs = tf.map_fn(self._simple_prob, batch)

        return probs


class MLPBlock(tfk.layers.Layer):
    def __init__(self, hidden_dims=(128, 128), batch_norm=False):
        super(MLPBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm

    def call(self, inputs):
        x = inputs
        for layer_dim in self.hidden_dims:
            x = tfkl.Dense(layer_dim, activation='relu')(x)
            if self.batch_norm:
                x = tfkl.BatchNormalization()(x)

        return x


class UniformCategoricalSkillModel(SkillModel):
    def __init__(self, input_dim, hidden_dims, latent_dim, batch_norm=False):
        super(UniformCategoricalSkillModel, self).__init__(input_dim, hidden_dims, latent_dim)
        mlp = MLPBlock(hidden_dims, batch_norm=batch_norm)

        inputs = tfkl.Input(shape=(input_dim,))
        x = mlp(inputs)
        z_enc = tfkl.Dense(latent_dim, activation='softmax')(x)

        self.model = tfk.Model(inputs, z_enc, name='discriminator')

        self.model.compile(
            loss=tfk.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def call(self, x, return_probs=False):
        probs = self.model.predict(x)
        if return_probs:
            return probs
        return tf.one_hot([tf.argmax(probs, axis=-1)], self.latent_dim)

    def log_prob(self, x, y, return_full=False):
        """expects z as one-hot vector"""
        probs = self.model.predict(x)

        if not return_full:
            probs = tf.reduce_max(tf.multiply(probs, y), axis=-1)

        return tf.math.log(probs)

    def save(self, log_dir):
        self.model.save(log_dir)


class GaussianSkillModel(SkillModel):
    def __init__(self, input_dim, hidden_dims, latent_dim, batch_norm=False, fix_variance=True):
        super(GaussianSkillModel, self).__init__(input_dim, hidden_dims, latent_dim)
        self.fix_variance = fix_variance

        #mlp = MLPBlock(hidden_dims, batch_norm)

        inputs = tfkl.Input(shape=(input_dim,))
        #x = mlp(inputs)
        x = inputs
        for layer_dim in hidden_dims:
            x = tfkl.Dense(layer_dim, activation='relu')(x)
            if batch_norm:
                x = tfkl.BatchNormalization()(x)

        if fix_variance:
            """learn only the mean and fix the variance to 1"""
            outputs = tfkl.Dense(self.latent_dim)(x)
            self.variance = 1.

            self.model = tfk.Model(inputs, outputs)

            def loss(y_true, y_pred):
                pred_distr = self.fixed_var_distribution(y_pred)
                return -1 * pred_distr.log_prob(y_true)

            self.model.compile(
                loss=loss,
                optimizer=tfk.optimizers.Adam(),
                metrics='accuracy'
            )
        else:
            params_size = tfpl.MixtureSameFamily.params_size(self.latent_dim, component_params_size=tfpl.IndependentNormal.params_size([1]))
            x = tfkl.Dense(params_size, activation=None)(x)
            outputs = tfpl.MixtureSameFamily(self.latent_dim, tfpl.IndependentNormal([1]))(x)

            self.model = tfk.Model(inputs, outputs, name='discriminator')

            self.model.compile(
                loss=lambda y, model: -model.log_prob(y),
                optimizer=tfk.optimizers.Adam(),
                metrics=["accuracy"]
            )

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def fixed_var_distribution(self, means):
        return tfd.MultivariateNormalDiag(loc=means, scale_diag=np.full(self.latent_dim, self.variance))

    def call(self, x):
        # I think this automatically samples from the distribution produced in the output layer,
        # rather than returning the distribution
        if self.fix_variance:
            pred = self.model.predict(x)
        else:
            pred_distr = self.fixed_var_distribution(self.model.predict(x))
            pred = pred_distr.sample()

        return pred

    def log_prob(self, x, y):
        distr = self.model(x)
        if self.fix_variance:
            distr = self.fixed_var_distribution(distr)

        return distr.log_prob(y)

    def save(self, log_dir):
        self.model.save(log_dir)
