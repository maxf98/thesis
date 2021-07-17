import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Input
import tensorflow.keras.backend as K
from layers import VAELayer, VQVAELayer, EncoderNetwork, MLPBlock, DecoderNetwork, vq_vae_loss_wrapper

tfd = tfp.distributions


class VAE(keras.Model):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        decoder_hidden_dims = (hidden_dims[-i] for i in range(len(hidden_dims)))  # invert encoder dims for decoder

        self.encoder = MLPBlock(hidden_dims)
        self.vae_layer = VAELayer(latent_dim, 0.5)
        self.decoder = DecoderNetwork(decoder_hidden_dims, input_dim)

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        z = self.vae_layer(x)
        x_hat = self.decoder(z)
        return x_hat





