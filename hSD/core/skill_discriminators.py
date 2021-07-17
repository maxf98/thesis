from abc import ABC, abstractmethod
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
    def train(self, x, z=None):
        """perform discriminator(/generator) training"""

    @abstractmethod
    def log_probs(self, x, z):
        """evaluate likelihood of mapping x to z under discriminator"""


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

    def log_probs(self, x, z, return_full=False):
        """
        expects z as one-hot vector
        """
        probs = self.model.predict(x)

        if not return_full:
            probs = tf.reduce_max(tf.multiply(probs, z), axis=-1)

        return tf.math.log(probs)


