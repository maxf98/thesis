from abc import ABC, abstractmethod

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory

from core.modules import utils
import tensorflow_probability as tfp



class RolloutDriver(ABC):
    def __init__(self,
                 environment: TFEnvironment,
                 policy: TFPolicy,
                 skill_prior: tfp.distributions.Distribution,
                 skill_length,
                 episode_length,
                 buffer_size):
        """fills the replay buffer with experience from the environment, collected using the given policy"""
        self.environment = environment
        self.policy = policy
        self.skill_prior = skill_prior
        self.skill_length = skill_length
        self.episode_length = episode_length
        self.replay_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=buffer_size)

    @abstractmethod
    def collect_experience(self, num_steps):
        pass


class BaseRolloutDriver(RolloutDriver):
    """collects simple transitions (s, z -> s') from the environment, with skill resampling"""
    def __init__(self,
                 environment,
                 policy,
                 skill_prior,
                 skill_length,
                 episode_length,
                 buffer_size
                 ):
        super().__init__(environment, policy, skill_prior, skill_length, episode_length, buffer_size)

    def collect_experience(self, num_steps):
        time_step = self.environment.reset()
        skill = self.skill_prior.sample()
        skill_i, episode_i = 0, 0
        cur_skill_transitions = []  # list of transitions

        for _ in range(num_steps):
            if skill_i == self.skill_length:
                skill_i = 0
                skill = self.skill_prior.sample()

            if episode_i == self.episode_length or time_step.is_last():
                skill_i, episode_i = 0, 0
                skill = self.skill_prior.sample()
                time_step = self.environment.reset()

            aug_ts = utils.aug_time_step(time_step, skill)
            action_step = self.policy.action(aug_ts)
            next_time_step = self.environment.step(action_step.action)
            next_aug_ts = utils.aug_time_step(next_time_step, skill)
            cur_skill_transitions.append(trajectory.from_transition(aug_ts, action_step, next_aug_ts))
            self.replay_buffer.add_batch(trajectory.from_transition(aug_ts, action_step, next_aug_ts))

            time_step = next_time_step
            skill_i += 1
            episode_i += 1


def collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length):
    trajectories = [[] for _ in range(len(skills))]

    for i in range(len(skills)):
        for si in range(rollouts_per_skill):
            time_step = env.reset()
            cur_traj = []  # only collects states
            for ti in range(skill_length):
                cur_traj.append(time_step.observation.numpy().flatten().tolist())
                aug_time_step = utils.aug_time_step(time_step, skills[i])
                action_step = policy.action(aug_time_step)
                time_step = env.step(action_step.action)

            trajectories[i].append(cur_traj)

    return trajectories

