import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist

tf.compat.v1.disable_v2_behavior()

# Load data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data.
x_test = x_test / np.max(x_train)
x_train = x_train / np.max(x_train)

# Add input channel dimension.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Target dictionary.
target_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# VQ layer.
class VQVAELayer(Layer):
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


# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance

        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss

        return recon_loss + loss  # * beta

    return vq_vae_loss




# Hyper Parameters.
epochs = 1000 # MAX
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
embedding_dim = 32 # Length of embedding vectors.
num_embeddings = 128 # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = 0.25 # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=True)




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

# Plot training results.
loss = history.history['loss'] # Training loss.
val_loss = history.history['val_loss'] # Validation loss.
num_epochs = range(1, 1 + len(history.history['loss'])) # Number of training epochs.

plt.figure(figsize=(16,9))
plt.plot(num_epochs, loss, label='Training loss') # Plot training loss.
plt.plot(num_epochs, val_loss, label='Validation loss') # Plot validation loss.

plt.title('Training and validation loss')
plt.legend(loc='best')
plt.show()


# Show original reconstruction.
n_rows = 5
n_cols = 8 # Must be divisible by 2.
samples_per_col = int(n_cols / 2)
sample_offset = np.random.randint(0, len(x_test) - n_rows * n_cols - 1)
#sample_offset = 0

img_idx = 0
plt.figure(figsize=(n_cols * 2, n_rows * 2))
for i in range(1, n_rows + 1):
    for j in range(1, n_cols + 1, 2):
        idx = n_cols * (i - 1) + j

        # Display original.
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.title.set_text('({:d}) Label: {:s} ->'.format(
            img_idx,
            str(target_dict[np.argmax(y_test[img_idx + sample_offset])])))
        ax.imshow(x_test[img_idx + sample_offset].reshape(28, 28),
                  cmap='gray_r',
                  clim=(0, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction.
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.title.set_text('({:d}) Reconstruction'.format(img_idx))
        ax.imshow(vqvae.predict(
            x_test[img_idx + sample_offset].reshape(-1, 28, 28, 1)).reshape(28, 28),
            cmap='gray_r',
            clim=(0, 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        img_idx += 1
plt.show()
