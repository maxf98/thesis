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


class VAESkillDiscriminator(SkillDiscriminator):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim):
        """performs skill discovery, i.e. mapping inputs to latents,
         either via unsupervised (VAE) or supervised (MLP) training"""
        super().__init__(input_dim, hidden_dim, latent_dim)

        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

        self.encoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=input_dim),
            tfkl.Dense(hidden_dim),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim),
                       activation=None),
            tfpl.MultivariateNormalTriL(
                latent_dim,
                activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
        ])

        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=[latent_dim]),
            tfkl.Dense(hidden_dim),
            tfkl.Dense(input_dim)
        ])

        self.vae = tfk.Model(inputs=encoder.inputs,
                        outputs=decoder(encoder.outputs[0]))

        negloglik = lambda x, rv_x: -rv_x.log_prob(x)

        self.vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                    loss=negloglik)

    def train(self, x, z=None):
        return self.vae.fit(x, x, epochs=100, batch_size=32)

    def log_probs(self, x, z):
        pass
