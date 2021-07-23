from abc import ABC, abstractmethod

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.typing.types import NestedTensor
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import TensorSpec

from core import utils
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from core.policies import FixedOptionPolicy


class RolloutDriver(ABC):
    def __init__(self,
                 environment: TFEnvironment,
                 policy: TFPolicy,
                 skill_prior: tfp.distributions.Distribution,
                 skill_length,
                 episode_length,
                 buffer_size,
                 preprocess_fn=None):
        """fills the replay buffer with experience from the environment, collected using the given policy"""
        self.environment = environment
        self.policy = policy
        self.skill_prior = skill_prior
        self.skill_length = skill_length
        self.episode_length = episode_length
        self.replay_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=buffer_size)
        self.preprocess_fn = preprocess_fn

    @abstractmethod
    def collect_experience(self, num_steps):
        pass


class BaseRolloutDriver(RolloutDriver):
    """collects simple transitions from the environment"""
    def __init__(self,
                 environment,
                 policy,
                 skill_prior,
                 skill_length,
                 episode_length,
                 buffer_size,
                 preprocess_fn=None  # function to apply to transitions before adding to the replay buffer (i.e. convert to data spec)
                 ):
        super().__init__(environment, policy, skill_prior, skill_length, episode_length, buffer_size, preprocess_fn)

    def collect_experience(self, num_steps):
        time_step = self.environment.reset()
        skill = self.skill_prior.sample()
        skill_i, episode_i = 0, 0
        cur_skill_transitions = []  # list of transitions

        for _ in tqdm(range(num_steps)):
            if skill_i == self.skill_length:  # preprocess skill trajectory to match replay_buffer data_spec and add
                if self.preprocess_fn is not None:
                    cur_skill_transitions = self.preprocess_fn(cur_skill_transitions)
                # self.replay_buffer.add_batch(cur_skill_transitions)
                skill_i = 0
                skill = self.skill_prior.sample()

            if episode_i == self.episode_length or time_step.is_last():
                skill_i, episode_i = 0, 0
                skill = self.skill_prior.sample()
                time_step = self.environment.reset()  # for now we don't include partial trajectories

            aug_ts = utils.aug_time_step(time_step, skill)
            action_step = self.policy.action(aug_ts)
            next_time_step = self.environment.step(action_step.action)
            next_aug_ts = utils.aug_time_step(next_time_step, skill)
            cur_skill_transitions.append(trajectory.from_transition(aug_ts, action_step, next_aug_ts))
            self.replay_buffer.add_batch(trajectory.from_transition(aug_ts, action_step, next_aug_ts))

            time_step = next_time_step
            skill_i += 1
            episode_i += 1


def collect_skill_trajectories(env, policy, skills, rollouts_per_skill, trajectory_length):
    trajectories = [[] for _ in range(len(skills))]

    for i in range(len(skills)):
        skill_policy = FixedOptionPolicy(policy, skills[i])
        for si in range(rollouts_per_skill):
            time_step = env.reset()
            cur_traj = []  # only collects states
            for ti in range(trajectory_length):
                cur_traj.append(time_step.observation.numpy().flatten().tolist())
                action_step = skill_policy.action(time_step)
                time_step = env.step(action_step.action)

            trajectories[i].append(cur_traj)

    return trajectories

