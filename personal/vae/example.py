from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from thesis.personal.vae import vae_mlp, mnist_plot_results

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.losses import mse

import numpy as np

tfd = tfp.distributions


def vae_mlp_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 50

    vae = vae_mlp.VAE(original_dim, intermediate_dim, latent_dim)

    vae.train(x_train, epochs, batch_size)

    models = (vae.encoder, vae.decoder)
    data = (x_test, y_test)

    mnist_plot_results.plot_results(models,
                     data,
                     batch_size=batch_size,
                     model_name="vae_mlp")


def vae_mlp_point_env():
    samples = sample_from_uniform_categorical(num_classes=10, dim=2, num_samples=2000)

    input_dim = 2
    hidden_dim = 128

    latent_dim = 1
    epochs = 100
    batch_size = 128

    vae = vae_mlp.TFPVAE(input_dim, hidden_dim, latent_dim)

    vae.train(samples, epochs, batch_size)

    eval = samples[:10]

    latent_samples = tfp.distributions.Normal(loc=0., scale=1.).sample(10)
    #latent_samples = tfp.distributions.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.]).sample(10)

    print(eval)
    z = vae.encoder(eval)
    print("mean: {} ––– variance: {}".format(tf.reduce_mean(z.mean()), tf.reduce_mean(z.variance())))

    print(np.around(vae.vae(eval), 2))

    print(np.around(vae.decoder(latent_samples), 2))


def sample_from_uniform_categorical(num_classes=10, dim=2, num_samples=2000):
    probs = [[1. / num_classes] * num_classes] * dim
    dist = tfd.Multinomial(total_count=1, probs=probs)
    one_hot_samples = dist.sample(num_samples)
    indexes = tf.argmax(one_hot_samples, axis=-1)
    float_indexes = tf.cast(indexes, dtype=tf.float64)
    normalized = tf.map_fn(lambda x: (x - 5) / 10, float_indexes)
    return normalized


if __name__ == '__main__':
    #print(mse([0.2, 0.2], [0.2, 0.1]))
    vae_mlp_point_env()