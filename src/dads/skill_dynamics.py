import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model


from skill_discriminator import SkillDiscriminator

L=10  # DADS: 500 (?? is it? can't find that anywhere in code...)

""" predicts state difference to next state, given the current state and skill """

class SkillDynamics(SkillDiscriminator):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 fc_layer_sizes=(64, 64)):

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        inputs = Input(shape=input_dim, name='encoder_input')
        x = Dense(fc_layer_sizes[0], activation='relu')(inputs)
        x = Dense(fc_layer_sizes[1], activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling,
                   output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])

        self.model = Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='discriminator')

        # computes mse to latent pred z (model outputs z_mean, z_log_var and z)
        def loss(y_true, y_pred):
            return keras.losses.mse(y_true, y_pred[2])

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )


    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling
            fr an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        # K is the keras backend
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, x, y):
        """ x = (s, z) and y = s'-s """
        # batch.observation.shape = 128 x 2 x 4 => 2 obs, 2 skill
        # get initial observations (state, skill)
        history = self.model.fit(x, y, 128, verbose=0)
        return history

    def call(self, batch):
        """ predict s' - s from s and z => expects single-step transitions """
        return self.model.predict(batch)

    def log_prob(self, batch):
        """ expects batch input dim as BATCH_SIZE x 2 x OBS_DIM
            for each in batch, compute p(s'|s, z) / sum(p(s'|s,z_i)) for L randomly sampled z_i
            # computes the internal reward for DADS"""
        x = batch[:, 0, :]
        y = tf.subtract(batch[:, 1, :-self.latent_dim], batch[:, 0, :-self.latent_dim])
        z_mean, z_log_var, _ = self.model.predict(x)

        latent_distributions = tfp.distributions.MultivariateNormalDiag(z_mean, K.exp(z_log_var))
        probs = latent_distributions.prob(y)
        x_obs = x[:,:-self.latent_dim]

        denoms = tf.stack([self._denom(x_obs[i], y[i]) for i in range(len(x))])

        normalized_probs = tf.math.log(tf.math.divide(probs, denoms))

        return normalized_probs


    def _denom(self, obs, obs_delta):
        skill_prior = tfp.distributions.Uniform([-1, -1], [1, 1])
        zi = skill_prior.sample(L)
        sampled_obs = tf.map_fn(lambda x: tf.concat([obs, x], -1), zi)
        z_mean, z_log_var, _ = self.model.predict(sampled_obs)

        latent_distributions = tfp.distributions.MultivariateNormalDiag(z_mean, K.exp(z_log_var))
        probs = latent_distributions.prob(obs_delta)

        d = (1/L) * tf.math.reduce_sum(probs)

        return d


    def save(self, save_to):
        self.model.save(save_to)
