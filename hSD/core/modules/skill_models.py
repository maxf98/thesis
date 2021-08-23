from abc import ABC, abstractmethod
import math
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
import tensorflow_probability as tfp
from tf_agents.distributions import tanh_bijector_stable

import numpy as np


class SkillModel(ABC):
    @abstractmethod
    def train(self, x, y=None, batch_size=32, epochs=1):
        """perform discriminator(/generator) training"""

    @abstractmethod
    def call(self, x):
        """return encoded (sampled) values"""

    @abstractmethod
    def log_prob(self, x, y):
        """evaluate likelihood of mapping x to z under discriminator"""

    @abstractmethod
    def save(self, log_dir):
        """saves the underlying discriminator model (not necessarily keras model, not necessarily need to be saved)"""


class SkillDiscriminator(SkillModel):
    def __init__(
            self,
            observation_dim,
            skill_dim,
            skill_type,
            normalize_observations=False,
            fc_layer_params=(256, 256),
            fix_variance=False):
        super(SkillDiscriminator, self).__init__()
        self.observation_dim = observation_dim
        self.skill_dim = skill_dim
        self.skill_type = skill_type
        self.normalize_observations = normalize_observations
        self.fc_layer_params = fc_layer_params
        self.fix_variance = fix_variance

        if not self.fix_variance:
            self.std_lower_clip = 0.3
            self.std_upper_clip = 10.0

        inputs = tfkl.Input(self.observation_dim)
        if self.normalize_observations:
            inputs = tfkl.BatchNormalization(inputs)

        outputs, prior_distribution = self.default_discriminator(inputs)

        self.model = tfk.Model(inputs, outputs, name='discriminator')

        self.model.compile(
            loss=lambda y, model: -model.log_prob(y),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def get_distributions(self, out):
        if self.skill_type in ['gaussian', 'cont_uniform']:
            mean = tfkl.Dense(
                out, self.skill_dim, name='mean')
            if not self.fix_variance:
                stddev = tf.clip_by_value(
                    tfkl.Dense(
                        out,
                        self.skill_dim,
                        activation=tf.nn.softplus,
                        name='stddev'), self.std_lower_clip, self.std_upper_clip)
            else:
                stddev = tf.fill([tf.shape(out)[0], self.skill_dim], 1.0)

            inference_distribution = tfd.MultivariateNormalDiag(
                loc=mean, scale_diag=stddev)

            if self.skill_type == 'gaussian':
                prior_distribution = tfd.MultivariateNormalDiag(
                    loc=[0.] * self.skill_dim, scale_diag=[1.] * self.skill_dim)
            elif self.skill_type == 'cont_uniform':
                prior_distribution = tfd.Independent(
                    tfd.Uniform(
                        low=[-1.] * self.skill_dim, high=[1.] * self.skill_dim),
                    reinterpreted_batch_ndims=1)

                # squash posterior to the right range of [-1, 1]
                bijectors = []
                bijectors.append(tanh_bijector_stable.Tanh())
                bijector_chain = tfp.bijectors.Chain(bijectors)
                inference_distribution = tfd.TransformedDistribution(
                    distribution=inference_distribution, bijector=bijector_chain)

        elif self.skill_type == 'discrete_uniform':
            logits = tfkl.Dense(
                out, self.skill_dim, name='logits')
            inference_distribution = tfd.OneHotCategorical(logits=logits)
            prior_distribution = tfd.OneHotCategorical(probs=[1. / self.skill_dim] * self.skill_dim)
        elif self.skill_type == 'multivariate_bernoulli':
            print('Not supported yet')

        return inference_distribution, prior_distribution

    def default_discriminator(self, inputs):
        out = inputs
        for idx, layer_size in enumerate(self.fc_layer_params):
            out = tfkl.Dense(
                out,
                layer_size,
                activation=tf.nn.relu,
                name='hid_' + str(idx))

        return self.get_distributions(out)

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def call(self, x, return_probs=False):
        probs = self.model.predict(x)
        if return_probs:
            return probs
        return tf.one_hot([tf.argmax(probs, axis=-1)], self.latent_dim)

    def log_prob(self, x, y, return_full=False):
        """expects z as one-hot vector"""
        probs = self.model.predict(x)

        if not return_full:
            probs = tf.reduce_max(tf.multiply(probs, y), axis=-1)

        return tf.math.log(probs)

    def save(self, log_dir):
        self.model.save(log_dir)


class SkillDynamics(SkillModel):
    def __init__(
            self,
            observation_dim,
            skill_dim,
            normalize_observations=False,
            fc_layer_params=(256, 256),
            network_type='default',
            num_components=1,
            fix_variance=False,
            reweigh_batches=False):

        self.observation_dim = observation_dim
        self.skill_dim = skill_dim
        self.normalize_observations = normalize_observations
        self.reweigh_batches = reweigh_batches

        # dynamics network properties
        self._fc_layer_params = fc_layer_params
        self._network_type = network_type
        self._num_components = num_components
        self._fix_variance = fix_variance
        if not self._fix_variance:
            self._std_lower_clip = 0.3
            self._std_upper_clip = 10.0

        inputs = tfkl.Input(observation_dim + skill_dim)
        outputs = self.default_model(inputs)

        self.model = tfk.Model(inputs, outputs, name="dynamics")
        self.model.compile(
            loss=lambda y, model: -model.log_prob(y),
            optimizer=tfk.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def get_distribution(self, out):
        if self._num_components > 1:
            self.logits = tfkl.Dense(
                out, self._num_components, name='logits')
            means, scale_diags = [], []
            for component_id in range(self._num_components):
                means.append(tfkl.Dense(out, self.observation_dim, name='mean_' + str(component_id)))
                if not self._fix_variance:
                    scale_diags.append(
                        tf.clip_by_value(
                            tfkl.Dense(
                                out,
                                self.observation_dim,
                                activation=tf.nn.softplus,
                                name='stddev_' + str(component_id)), self._std_lower_clip, self._std_upper_clip))
                else:
                    scale_diags.append(tf.fill([tf.shape(out)[0], self.observation_dim], 1.0))

            self.means = tf.stack(means, axis=1)
            self.scale_diags = tf.stack(scale_diags, axis=1)
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    logits=self.logits),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=self.means, scale_diag=self.scale_diags))

        else:
            mean = tf.compat.v1.layers.dense(
                out, self.observation_dim, name='mean', reuse=tf.compat.v1.AUTO_REUSE)
            if not self._fix_variance:
                stddev = tf.clip_by_value(
                    tf.compat.v1.layers.dense(
                        out,
                        self.observation_dim,
                        activation=tf.nn.softplus,
                        name='stddev',
                        reuse=tf.compat.v1.AUTO_REUSE), self._std_lower_clip,
                    self._std_upper_clip)
            else:
                stddev = tf.fill([tf.shape(out)[0], self.observation_dim], 1.0)
            return tfd.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

    def default_model(self, inputs):
        out = inputs
        for idx, layer_size in enumerate(self._fc_layer_params):
            out = tf.compat.v1.layers.dense(
                out,
                layer_size,
                activation=tf.nn.relu,
                name='hid_' + str(idx),
                reuse=tf.compat.v1.AUTO_REUSE)

        return self.get_distribution(out)

    def train(self, x, y=None, batch_size=32, epochs=1):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def call(self, x):
        # I think this automatically samples from the distribution produced in the output layer,
        # rather than returning the distribution
        return self.model.predict(x)

    def log_prob(self, x, y):
        distr = self.model(x)
        return distr.log_prob(y)

    def save(self, log_dir):
        self.model.save(log_dir)