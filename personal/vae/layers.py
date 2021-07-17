import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Input
import tensorflow.keras.backend as K

tfd = tfp.distributions


class MLPBlock(Layer):
    def __init__(self, hidden_dims=(128, 128)):
        super(MLPBlock, self).__init__()
        self.layers = [Dense(d, activation='relu') for d in hidden_dims]

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderNetwork(Layer):
    def __init__(self, input_dim, hidden_dims):
        super(EncoderNetwork, self).__init__()

        self.layers = [Input(shape=(input_dim, ))] + [Dense(d, activation='relu') for d in hidden_dims]

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderNetwork(Layer):
    def __init__(self, hidden_dims, input_dim):
        super(DecoderNetwork, self).__init__()
        self.layers = [Dense(d, activation='relu') for d in hidden_dims] + [Dense(input_dim, activation='linear')]

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x



class VAELayer(Layer):
    # implements the reparameterization trick and adds kl loss to embedding
    def __init__(self, latent_dim, kl_weight):
        super(VAELayer, self).__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    def call(self, inputs):
        x = Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim), activation=None)(inputs)
        z = tfp.layers.MultivariateNormalTriL(self.latent_dim,
                                              activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior, weight=self.kl_weight))(x)
        return z


class VQVAELayer(Layer):
    """
    ::: EXAMPLE USAGE :::
        # Encoder
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        x = Dense(512, activation='relu')(x)
        x = Dense(embedding_dim, activation='relu')(x)
        enc_inputs = x
        enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(x)
        x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)
        data_variance = np.var(x_train)
        loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)

        # Decoder.
        x = Dense(512, activation='relu')(x)
        x = Dense(28*28)(x)
        x = tf.keras.layers.Reshape((28, 28, 1))(x)

        # Autoencoder.
        vqvae = Model(input_img, x)
        vqvae.compile(loss=loss, optimizer='adam')
        vqvae.summary()

        history = vqvae.fit(x_train, x_train,
                            batch_size=batch_size, epochs=epochs,
                            validation_split=validation_split,
                            callbacks=[esc])

    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                 shape=(self.embedding_dim, self.num_embeddings),
                                 initializer=self.initializer,
                                 trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, inputs):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(inputs, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs ** 2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(inputs)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        # avg_probs = K.mean(encodings, axis=0)
        # perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))

        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices)


def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = keras.losses.mse(x, x_hat) / data_variance

        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss

        return recon_loss + loss  # * beta

    return vq_vae_loss