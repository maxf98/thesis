import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

# TODO: change network output to multivariate Gaussian with diagonal covariance matrix
class Discriminator:
    def __init__(self,
                 input_dim,
                 intermediate_dim,
                 latent_dim):
        self.input_dim, self.latent_dim = input_dim, latent_dim

        inputs = Input(shape=input_dim, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        self.model = Model(inputs=inputs, outputs=[z_mean, z_log_var], name='discriminator')

        def loss(y_true, y_pred):
            distr = tfp.distributions.MultivariateNormalDiag(y_pred[0], K.exp(y_pred[1]))
            negative_log_probs = -1 * distr.log_prob(y_true)
            return negative_log_probs

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        self.model.summary()

    def train(self, x, y):
        return self.model.fit(x, y, verbose=1)

    def _split_batch(self, batch):
        # could time the split operation, it's technically avoidable so if time consuming change replay buffer
        x, y = tf.split(batch.observation, [batch.observation.shape[2] - self.latent_dim, self.latent_dim], 2)
        return x, y

    # expects unaugmented observations
    def call(self, batch):
        x = tf.reshape(batch, [batch.shape[0], -1])
        return self.model.predict(x)

    def log_prob(self, sg, z):
        """
        compute probability of ground truth z given s and g (concatenated)
        """
        z_mean, z_log_var = self.model.predict(sg)
        latent_distributions = tfp.distributions.MultivariateNormalDiag(z_mean, K.exp(z_log_var))
        log_probs = latent_distributions.log_prob(z)

        return log_probs

    def save(self, save_to):
        self.model.save(save_to)