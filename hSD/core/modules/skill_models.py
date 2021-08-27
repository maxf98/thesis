from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
import tensorflow_probability as tfp
from tf_agents.distributions import tanh_bijector_stable

import numpy as np


class SkillModel(ABC):
    @abstractmethod
    def train(self, x, y=None, batch_size=32, epochs=1):
        """perform model(/generator) training"""

    @abstractmethod
    def call(self, x):
        """return encoded (sampled) values"""

    @abstractmethod
    def log_prob(self, x, y):
        """evaluate likelihood of mapping x to z under current model"""

    @abstractmethod
    def save(self, log_dir):
        """saves the underlying skill model (not necessarily keras model, not necessarily need to be saved)"""


"""
Base probabilistic skill model...
DADS implements a mixture of experts model, we can do that too still, but for point environments this should suffice
anyway...
"""
class BaseSkillModel(SkillModel):
    def __init__(
            self,
            input_dim,
            output_dim,
            skill_type,
            normalize_observations=False,
            fc_layer_params=(256, 256),
            fix_variance=False):
        super(BaseSkillModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skill_type = skill_type
        self.normalize_observations = normalize_observations
        self.fc_layer_params = fc_layer_params
        self.fix_variance = fix_variance

        if not self.fix_variance:
            self.std_lower_clip = 0.3
            self.std_upper_clip = 10.0

        inputs = tfkl.Input(self.input_dim)
        outputs = self.build_model(inputs)

        self.model = tfk.Model(inputs, outputs, name='discriminator')

        self.model.compile(
            loss=lambda y, p_y: -p_y.log_prob(y),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def build_model(self, inputs):
        x = tfkl.BatchNormalization(inputs) if self.normalize_observations else inputs
        x = self.mlp_block(x)
        return self.apply_distribution(x)

    def apply_distribution(self, x):
        if self.skill_type in ['gaussian', 'cont_uniform']:
            if self.fix_variance:
                x = tfkl.Dense(self.output_dim)(x)
                x = tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=[0.1]*self.output_dim))(x)
            else:
                x = tfkl.Dense(tfpl.IndependentNormal.params_size(self.output_dim))(x)  # top half learns means, bottom half learns variance
                x = tfpl.IndependentNormal(self.output_dim)(x)  # no variance clipping yet
                """
                x = tfkl.Dense(2 * self.output_dim)(x)
                x = tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
                    loc=t[:self.output_dim],
                    scale_diag=tf.clip_by_value(t[self.output_dim:], self.std_lower_clip, self.std_upper_clip)
                ))(x)
                """
            if self.skill_type == 'cont_uniform':
                # squash posterior to the right range of [-1, 1]
                bijector_chain = tfp.bijectors.Chain([tanh_bijector_stable.Tanh()])  # not really sure what this is tbh
                x = tfpl.DistributionLambda(lambda t: tfd.TransformedDistribution(
                    distribution=t, bijector=bijector_chain))(x)

        elif self.skill_type == 'discrete_uniform':
            x = tfk.layers.Dense(tfpl.OneHotCategorical.params_size(self.output_dim) - 1)(x)
            x = tfk.layers.Lambda(lambda x: tf.pad(x, paddings=[[0, 0], [1, 0]]))(x)
            x = tfpl.OneHotCategorical(self.output_dim)(x)

        return x

    def mlp_block(self, inputs):
        x = inputs
        for idx, layer_size in enumerate(self.fc_layer_params):
            x = tfkl.Dense(
                layer_size,
                activation=tf.nn.relu,
                name='hid_' + str(idx))(x)

        return x

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

    def call(self, x):
        return self.model(x)

    def log_prob(self, x, y, return_full=False):
        """expects z as one-hot vector"""
        pred_distr = self.model(x)
        return pred_distr.log_prob(y)

    def save(self, log_dir):
        self.model.save(log_dir)


"""implements mixture of experts skill dynamics model used in DADS"""
class SkillDynamics(SkillModel):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_components=4,
            normalize_observations=False,
            fc_layer_params=(256, 256),
            fix_variance=False):
        super(SkillDynamics, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_components = num_components
        self.normalize_observations = normalize_observations
        self.fc_layer_params = fc_layer_params
        self.fix_variance = fix_variance

        if not self.fix_variance:
            self.std_lower_clip = 0.3
            self.std_upper_clip = 10.0

        inputs = tfkl.Input(self.input_dim)
        outputs = self.build_model(inputs)

        self.model = tfk.Model(inputs, outputs, name='discriminator')

        self.model.compile(
            loss=lambda y, p_y: -p_y.log_prob(y),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def build_model(self, inputs):
        x = tfkl.BatchNormalization(inputs) if self.normalize_observations else inputs
        x = self.mlp_block(x)
        return self.apply_distribution(x)

    def mlp_block(self, inputs):
        x = inputs
        for idx, layer_size in enumerate(self.fc_layer_params):
            x = tfkl.Dense(
                layer_size,
                activation=tf.nn.relu,
                name='hid_' + str(idx))(x)

        return x

    def apply_distribution(self, x):
        """for now we assume that the number of components is greater than 1 and the variance is fixed, throw an error otherwise..."""
        if not self.fix_variance:
            raise NotImplementedError()
        x = tfkl.Dense(tfpl.MixtureSameFamily.params_size(self.num_components, self.output_dim))(x)
        scale_diag = [0.5] * self.output_dim
        make_distr_func = lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=scale_diag)
        x = tfpl.MixtureSameFamily(self.num_components, tfpl.DistributionLambda(make_distr_func))(x)

        return x

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

    def call(self, x):
        return self.model(x)

    def log_prob(self, x, y, return_full=False):
        """expects z as one-hot vector"""
        pred_distr = self.model(x)
        return pred_distr.log_prob(y)

    def save(self, log_dir):
        self.model.save(log_dir)
