from abc import ABC, abstractmethod
import gin

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents import trajectories
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
        skills = [self.skill_prior.sample(self.episode_length // self.skill_length) for _ in range(num_steps // self.episode_length)]
        traj_observers = [self.online_buffer.add_batch, self.offline_buffer.add_batch]
        collect_experience_for_skills(self.environment, self.policy, preprocess_time_step, traj_observers, skills, self.skill_length,s_norm=self.state_norm)


def collect_experience_for_skills(env, policy, preprocess_fn, traj_observer, skills, skill_length,
                                  return_aug_obs=True, s_norm=True, render=False):
    """skills is a 2d array, each array contains skills to be executed within one episode
    after the episode, reset environment and run next skill sequence"""
    time_step = env.reset()

    try:
        _ = iter(policy)
    except TypeError:
        policy, skill_length, traj_observer = [policy], [skill_length], [traj_observer]

    for episode_i in range(len(skills)):
        for skill_i in range(len(skills[episode_i])):
            skill = skills[episode_i][skill_i]

            rollout_hier_skill_trajectory(time_step, env, policy, preprocess_fn, traj_observer, skill, skill_length, return_aug_obs=return_aug_obs, s_norm=s_norm, render=render)
        time_step = env.reset()


def rollout_skill_trajectory(time_step, env, policy, preprocess_fn, traj_observer, skill, skill_length,
                             return_aug_obs=True, s_norm=False, render=False):

    s_0 = utils.hide_goal(time_step.observation) if s_norm else None

    for i in range(skill_length):
        if render:
            env.render(mode='human')

        aug_ts = preprocess_fn(time_step, skill, s_0)
        action_step = policy.action(aug_ts)

        next_time_step = env.step(action_step.action)

        next_aug_ts = preprocess_fn(next_time_step, skill, s_0)

        if return_aug_obs:
            traj = trajectories.trajectory.from_transition(aug_ts, action_step, next_aug_ts)
        else:
            traj = trajectories.trajectory.from_transition(time_step, action_step, next_time_step)

        for fn in traj_observer:
            fn(traj)

        time_step = next_time_step

    return time_step


def preprocess_time_step(time_step, skill, s_norm):
    obs = utils.hide_goal(time_step.observation)
    obs = obs - s_norm if s_norm is not None else obs
    obs = tf.cast(obs, tf.float32)
    skill_shape = (1, -1) if len(obs.shape.as_list()) == 2 else (-1)  # distinguish between batched and unbatched environments
    obs = tf.concat([obs, tf.reshape(skill, skill_shape)], axis=-1)
    return ts.TimeStep(time_step.step_type,
                       time_step.reward,
                       time_step.discount,
                       obs)


def rollout_hier_skill_trajectory(time_step, env, policies, preprocess_fn, traj_observer, skill, skill_lengths,
                                  return_aug_obs=True, s_norm=True, render=False):
    """collect experience in the base environment given a hierarchy of policies
    descend to base_env_policy to choose skill and rollout for T steps
    recursive method
    policies, traj_observer and skill_length are layer-specific, given in ascending order
    """
    s_0 = utils.hide_goal(time_step.observation) if s_norm else None
    policy, skill_length = policies[-1], skill_lengths[-1]

    for i in range(skill_length):

        aug_ts = preprocess_fn(time_step, skill, s_0)
        action_step = policy.action(aug_ts)

        if len(policies) > 1:
            next_time_step = rollout_hier_skill_trajectory(time_step, env, policies[:-1], preprocess_fn,
                                                           traj_observer[:-1], action_step.action, skill_lengths[:-1],
                                                           return_aug_obs=return_aug_obs, s_norm=s_norm, render=render)
        else:
            if render:
                env.render(mode='human')

            next_time_step = env.step(action_step.action)

        if return_aug_obs:
            next_aug_ts = preprocess_fn(next_time_step, skill, s_0)
            traj = trajectories.trajectory.from_transition(aug_ts, action_step, next_aug_ts)
        else:
            traj = trajectories.trajectory.from_transition(time_step, action_step, next_time_step)

        for fn in traj_observer[-1]:
            fn(traj)

        time_step = next_time_step

    return time_step





