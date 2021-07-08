from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

tfd = tfp.distributions

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling,
                   output_shape=(self.latent_dim,),
                   name='z')([z_mean, z_log_var])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.hidden_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        #reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss = mse(inputs, outputs)

        reconstruction_loss *= input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        self.vae.summary()

    def train(self, x, epochs, batch_size):
        self.vae.fit(x, x, epochs=epochs, batch_size=batch_size)

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
        epsilon = K.random_normal(shape=(batch, dim), mean=0, stddev=1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class TFPVAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim), activation=None)(x)
        z = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior, weight=2))(x)

        self.encoder = Model(inputs, z, name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.hidden_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.input_dim, activation='tanh')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder.outputs[0])
        self.vae = Model(inputs, outputs, name='vae_mlp')
        self.vae.compile(optimizer='adam', loss=mse)
        self.vae.summary()

    def train(self, x, epochs, batch_size):
        val_split = 0.1
        esc = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                            patience=5, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=True)
        self.vae.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[])


class AutoEncoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bce = False

        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        z = Dense(self.latent_dim, name='z_mean')(x)

        self.encoder = Model(inputs, z, name='encoder')
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.hidden_dim, activation='relu')(latent_inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs))
        self.ae = Model(inputs, outputs, name='vae_mlp')
        self.ae.compile(optimizer='adam', loss=mse)
        self.ae.summary()

    def train(self, x, epochs, batch_size):
        self.ae.fit(x, x, epochs=epochs, batch_size=batch_size)