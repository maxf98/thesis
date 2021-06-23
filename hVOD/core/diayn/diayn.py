import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from skill_discovery import SkillDiscovery
from core import utils
from tqdm import tqdm
import math


class DIAYNAgent(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_discriminator,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 logger,
                 num_skills,
                 max_skill_length=50,
                 train_batch_size=128
                 ):
        super().__init__(train_env, eval_env, skill_discriminator, rl_agent, replay_buffer, logger, max_skill_length)

        rl_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_batch_size,
            num_steps=2).prefetch(3)
        discriminator_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_batch_size,
            num_steps=2).prefetch(3)
        self._train_batch_iterator = iter(rl_dataset)
        self._discriminator_batch_iterator = iter(discriminator_dataset)

        self.num_skills = num_skills
        self._skill_prior = tfp.distributions.Categorical(probs=[1 / num_skills for _ in range(num_skills)])

        self.exploration_rollouts = []

    def _rl_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return self._relabel_ir(experience)

    def _discriminator_training_batch(self):
        return next(self._discriminator_batch_iterator)

    def _relabel_ir(self, batch):
        obs_l, obs_r = tf.split(batch.observation, [1, 1], axis=1)
        obs_l, z = tf.split(obs_l, [-1, self.num_skills], axis=2)
        obs_l = tf.reshape(obs_l, [obs_l.shape[0], -1])
        z = tf.reshape(z, [obs_l.shape[0], -1])
        obs_r, _ = tf.split(obs_r, [-1, self.num_skills], axis=2)
        obs_r = tf.reshape(obs_r, [obs_l.shape[0], -1])

        # use if we wanna encode state delta instead of states (as in DIAYN)
        # obs_delta = tf.subtract(obs_r, obs_l)

        def new_rewards(x, y):
            pred_y = self.skill_discriminator.call(x)

            probs = tf.reduce_max(tf.multiply(pred_y, y), axis=1)
            r = tf.subtract(tf.math.log(probs), tf.math.log(1 / self.num_skills))  # add log(p(z)) term

            return r

        r = tf.stack([new_rewards(obs_l, z), new_rewards(obs_r, z)], axis=1)

        ret = batch.replace(reward=r)
        return ret

    def _collect_env(self, steps):
        time_step = self.train_env.reset()
        z = self._skill_prior.sample().numpy()
        z_one_hot = tf.one_hot([z], self.num_skills)
        path_length = 0
        for step in tqdm(range(steps)):
            if time_step.is_last() or path_length > self.max_skill_length:
                z = self._skill_prior.sample().numpy()
                z_one_hot = tf.one_hot([z], self.num_skills)
                time_step = self.train_env.reset()
                path_length = 0

            aug_time_step = utils.aug_time_step(time_step, z_one_hot)
            action_step = self.rl_agent.collect_policy.action(aug_time_step)
            next_time_step = self.train_env.step(action_step.action)
            aug_next_time_step = utils.aug_time_step(next_time_step, z_one_hot)

            self.exploration_rollouts.append(time_step.observation.numpy().flatten())

            traj = trajectory.from_transition(aug_time_step, action_step, aug_next_time_step)
            self.replay_buffer.add_batch(traj)

            time_step = next_time_step
            path_length += 1

    def _train_discriminator(self):
        experience, _ = self._discriminator_training_batch()
        t1, t2 = experience.observation[:, 0, :-self.num_skills], experience.observation[:, 1, :-self.num_skills]
        # delta_o = tf.subtract(t2, t1)
        z = experience.observation[:, 0, -self.num_skills:]
        discrim_history = self.skill_discriminator.train(t1, z)
        return discrim_history.history['loss'], discrim_history.history['accuracy']

    def _train_agent(self):
        experience = self._rl_training_batch()
        sac_loss = self.rl_agent.train(experience)

        # return loss
        return sac_loss.loss.numpy()

    def _log_epoch(self, epoch, discrim_stats, sac_stats):
        # find other ways of inspecting the discriminator => e.g. sampling skills and evaluating outputs
        self.logger.log(epoch, self.rl_agent.policy, self.skill_discriminator, self.eval_env, self.num_skills, discrim_stats, sac_stats, self.exploration_rollouts)
        self.exploration_rollouts = []


class DIAYNDiscriminator:
    def __init__(self,
                 input_shape,
                 intermediate_dim,
                 num_skills,
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

    def train(self, x, y):
        return self.model.fit(x, y, batch_size=128, verbose=0)

    def _split_batch(self, batch):
        x, y = tf.split(batch.observation, [batch.observation.shape[2] - self.num_skills, self.num_skills], 2)
        return x, y

    # expects unaugmented observations
    def call(self, batch):
        x = tf.reshape(batch, [batch.shape[0], -1])
        return self.model.predict(x)

    def save(self, save_to):
        self.model.save(save_to)


class OracleDiscriminator:
    def __init__(self, input_shape, intermediate_dim, num_skills, load_from=None):
        # params immediately discarded, only to make interchangeable with DIAYNDiscrim
        self.num_skills = num_skills  # for now we assume it's 4

    def train(self, x, y):
        pass

    def _prob(self, point):
        x, y = point
        msp = 0.001

        if x >= 0:
            if y >= 0:
                ret = [y / (x + y), x / (x + y), msp, msp]
            else:
                ret = [msp, x / (x - y), -y / (x - y), msp]
        else:
            if y >= 0:
                ret = [y / (y - x), msp, msp, -x / (y - x)]
            else:
                ret = [msp, msp, y / (x + y), x / (x + y)]

        if math.fabs(x) < msp and math.fabs(y) < msp:
            ret = [0.25, 0.25, 0.25, 0.25]

        dist_scaling_factor = 1 + (x**2 + y **2)

        return ret * dist_scaling_factor

    def _simple_prob(self, point):
        x, y = point
        msp = 0.001
        return (1 + 10 *(x**2 + y **2)) * np.array([1-msp if x>=0 and y>=0 else msp,
                         1-msp if x>=0 and y<0 else msp,
                         1-msp if x<0 and y<0 else msp,
                         1-msp if x<0 and y>=0 else msp])


    def call(self, batch):
        """
        Parameters
        ----------
        batch – batch expected as batch_size * 2

        Returns
        prob of each skill for each data_point in batch_size * 4
        """
        probs = tf.map_fn(self._simple_prob, batch)

        return probs
