import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

L = 100

class Discriminator:
    def __init__(self,
                 input_dim,
                 intermediate_dim,
                 latent_dim):
        self.input_dim, self.latent_dim = input_dim, latent_dim
        """
        we fix the variance at 0.1 for now... though I'm not sure what would be more reasonable
        """

        inputs = Input(shape=input_dim, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)

        self.model = Model(inputs=inputs, outputs=z_mean, name='discriminator')

        def loss(y_true, y_pred):  # maximise likelihood of samples under q_phi
            distr = tfp.distributions.MultivariateNormalDiag(y_pred, [1, 1])
            negative_log_probs = -1 * distr.log_prob(y_true)
            return negative_log_probs

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        self.model.summary()

    def train(self, x, y):
        return self.model.fit(x, y, verbose=0)

    # expects unaugmented observations
    def call(self, batch):
        x = tf.reshape(batch, [batch.shape[0], -1])
        return self.model.predict(x)

    def log_prob(self, sg, z):
        """
        compute probability of ground truth z given s and g (concatenated)
        """
        z_mean = self.model.predict(sg)
        latent_distributions = tfp.distributions.MultivariateNormalDiag(z_mean, [1, 1])
        probs = latent_distributions.log_prob(z)
        # denoms = tf.map_fn(self._denom, )

        # sample other z and

        return probs


    def save(self, save_to):
        self.model.save(save_to)
