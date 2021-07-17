from abc import ABC, abstractmethod

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.typing.types import NestedTensor
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import TensorSpec


class RolloutDriver(ABC):
    def __init__(self,
                 environment: TFEnvironment,
                 policy: TFPolicy,
                 skill_length,
                 episode_length,
                 buffer_size,
                 preprocess_fn=None):
        """fills the replay buffer with experience from the environment, collected using the given policy"""
        self.environment = environment
        self.policy = policy
        self.skill_length = skill_length
        self.episode_length = episode_length
        self.replay_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=buffer_size)
        self.preprocess_fn = preprocess_fn

    @abstractmethod
    def collect_experience(self, num_steps):
        pass

    @abstractmethod
    def get_experience(self) -> NestedTensor:
        pass


class BaseRolloutDriver(RolloutDriver):
    """collects simple transitions from the environment"""
    def __init__(self,
                 environment,
                 policy,
                 skill_length,
                 episode_length,
                 buffer_size,
                 preprocess_fn=None  # function to apply to transitions before adding to the replay buffer (i.e. convert to data spec)
                 ):
        super().__init__(environment, policy, skill_length, episode_length, buffer_size, preprocess_fn)

    def collect_experience(self, num_steps):
        time_step = self.environment.reset()
        skill_i, episode_i = 0, 0
        cur_skill_transitions = []  # list of transitions

        for _ in range(num_steps):
            if skill_i == self.skill_length:  # preprocess skill trajectory to match replay_buffer data_spec and add
                self.replay_buffer.add_batch(self.preprocess_fn(cur_skill_transitions))
                skill_i = 0

            if episode_i == self.episode_length or time_step.is_last():
                skill_i, episode_i = 0, 0
                time_step = self.environment.reset()  # for now we don't include partial trajectories

            action_step = self.policy.action(time_step)
            next_time_step = self.environment.step(action_step.action)
            cur_skill_transitions.append(trajectory.from_transition(time_step, action_step, next_time_step))

            time_step = next_time_step
            skill_i += 1
            episode_i += 1

    def get_experience(self, num_steps=None):
        dataset = self.replay_buffer.as_dataset(num_steps=num_steps, single_deterministic_pass=True)
        # but this could become relevant at some point
        iterator = iter(dataset)
        experience, _ = next(iterator)
        return experience


