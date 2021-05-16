import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# hyperparameters
intermediate_dim = 128
batch_size = 256


class DIAYNDiscriminator:
    def __init__(self,
                 num_skills,
                 input_shape=None,
                 load_from=None,
                 ):
        if load_from is None:
            self.num_skills = num_skills
            inputs = Input(shape=input_shape, name='encoder_input')
            x = Dense(intermediate_dim, activation='relu')(inputs)
            x = Dense(intermediate_dim, activation='relu')(x)
            skill_encoding = Dense(num_skills, activation='softmax')(x)
            self.model = Model(inputs=inputs, outputs=skill_encoding, name='discriminator')

            self.model.compile(
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"],
            )
        else:
            self.num_skills = num_skills
            self.model = keras.models.load_model(load_from)

        self.model.summary()

    # batch is single time steps (because diayn discriminator encodes states...)
    """
    def train(self, batch):
        # reformat batch for supervised training
        x, y = self._split_batch(batch)
        x = tf.reshape(x, [x.shape[0], -1])
        y = tf.reshape(y, [y.shape[0], -1])
        history = self.model.fit(x, y, batch_size=batch_size, validation_split=0.2, verbose=0)
        return history
    """
    def train(self, x, y):
        return self.model.fit(x, y, verbose=0)

    def _split_batch(self, batch):
        x, y = tf.split(batch.observation, [batch.observation.shape[2] - self.num_skills, self.num_skills], 2)
        return x, y

    # expects unaugmented observations
    def call(self, batch):
        x = tf.reshape(batch, [batch.shape[0], -1])
        return self.model.predict(x)

    def save(self, save_to):
        self.model.save(save_to)