import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy


# hyperparameters
intermediate_dim = 32
batch_size = 128
epochs = 4


class DIAYNDiscriminator():
    def __init__(self,
                 input_shape,
                 num_skills,
                 ):
        self.num_skills = num_skills
        inputs = Input(shape=input_shape, name='encoder input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        skill_encoding = Dense(num_skills, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=skill_encoding, name='discriminator')

        self.model.summary()
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

    # batch is single time steps (because diayn discriminator encodes states...)
    def train(self, batch):
        # reformat batch for supervised training
        x, y = self._split_batch(batch)
        x = tf.reshape(x, [x.shape[0], -1])
        y = tf.reshape(y, [y.shape[0], -1])
        history = self.model.fit(x, y, batch_size=256, epochs=epochs, validation_split=0.2)
        return history

    def _split_batch(self, batch):
        x, y = tf.split(batch.observation, [batch.observation.shape[2] - self.num_skills, self.num_skills], 2)
        return x, y

    # expects unaugmented observations
    def call(self, batch):
        x = tf.reshape(batch, [batch.shape[0], -1])
        return self.model.predict(x)