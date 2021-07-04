from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import argparse
import os

from skill_discriminator import SkillDiscriminator

"""
we use a multivariate gaussian distribution with diagonal covariance matrix
q_phi(z | delta_s) = Normal(mu_phi, sigma_phi)
"""
class VAEDiscriminator:
    def __init__(self,
                 input_dim,
                 intermediate_dim,
                 latent_dim):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary
        # with the TensorFlow backend
        z = Lambda(self.sampling,
                   output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = mse(inputs, outputs)

        reconstruction_loss *= input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        self.vae.summary()

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self, args):
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

    def train(self, delta_o):
        # could split input to use validation set
        return self.vae.fit(delta_o, validation_split=0.2)

    def encode(self, batch):
        return self.encoder.predict(batch)

    def decode(self, latents):
        return self.decoder.predict(latents)

    def log_probs(self, delta_s, z):
        z_mean, z_log_var, _ = self.encode(delta_s)
        latent_distributions = tfp.distributions.MultivariateNormalDiag(z_mean, K.exp(z_log_var))
        probs = latent_distributions.prob(z)

        return probs  # tf.math.log(probs)

    def save(self, path):
        self.encoder.save(os.path.join(path, "encoder"))
        self.decoder.save(os.path.join(path, "decoder"))