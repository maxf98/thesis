from abc import ABC, abstractmethod
import gin

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts


from core.modules import utils
import tensorflow_probability as tfp
import tensorflow as tf


class RolloutDriver(ABC):
    def __init__(self,
                 environment: TFEnvironment,
                 policy: TFPolicy,
                 skill_prior: tfp.distributions.Distribution,
                 skill_length,
                 episode_length,
                 online_buffer_size,
                 offline_buffer_size):
        """fills the replay buffer with experience from the environment, collected using the given policy"""
        self.environment = environment
        self.policy = policy
        self.skill_prior = skill_prior
        self.skill_length = skill_length
        self.episode_length = episode_length
        self.online_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=online_buffer_size)
        self.offline_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=offline_buffer_size)

    @abstractmethod
    def collect_experience(self, num_steps):
        pass


@gin.configurable
class BaseRolloutDriver(RolloutDriver):
    """collects simple transitions (s, z -> s') from the environment, with skill resampling"""
    def __init__(self,
                 environment,
                 policy,
                 skill_prior,
                 skill_length=100,
                 episode_length=100,
                 online_buffer_size=1000,
                 offline_buffer_size=10000,
                 state_norm=False
                 ):
        super().__init__(environment, policy, skill_prior, skill_length, episode_length, online_buffer_size, offline_buffer_size)
        self.state_norm = state_norm

    def collect_experience(self, num_steps):
        time_step = self.environment.reset()
        s_0 = time_step.observation
        skill = self.skill_prior.sample()
        skill_i, episode_i = 0, 0

        for _ in range(num_steps):
            if skill_i == self.skill_length:
                skill_i = 0
                skill = self.skill_prior.sample()
                s_0 = time_step.observation

            if episode_i == self.episode_length or time_step.is_last():
                skill_i, episode_i = 0, 0
                skill = self.skill_prior.sample()
                time_step = self.environment.reset()
                s_0 = time_step.observation

            aug_ts = self.preprocess_time_step(time_step, skill, s_0)
            action_step = self.policy.action(aug_ts)
            next_time_step = self.environment.step(action_step.action)
            next_aug_ts = self.preprocess_time_step(next_time_step, skill, s_0)
            self.online_buffer.add_batch(trajectory.from_transition(aug_ts, action_step, next_aug_ts))
            self.offline_buffer.add_batch(trajectory.from_transition(aug_ts, action_step, next_aug_ts))

            time_step = next_time_step
            skill_i += 1
            episode_i += 1

    def preprocess_time_step(self, time_step, skill, s_norm):
        obs = time_step.observation - s_norm if self.state_norm else time_step.observation
        obs = tf.cast(obs, tf.float32)
        return ts.TimeStep(time_step.step_type,
                           time_step.reward,
                           time_step.discount,
                           tf.concat([obs, tf.reshape(skill, (1, -1))], axis=-1))



