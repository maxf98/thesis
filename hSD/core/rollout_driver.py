from abc import ABC, abstractmethod

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.typing.types import NestedTensor


class RolloutDriver(ABC):
    def __init__(self,
                 environment: TFEnvironment,
                 policy: TFPolicy,
                 num_steps):
        """fills the replay buffer with experience from the environment, collected using the given policy"""
        self.environment = environment
        self.policy = policy
        self.num_steps = num_steps
        self.replay_buffer = TFUniformReplayBuffer(policy.collect_data_spec, environment.batch_size, max_length=num_steps)

    def set_policy(self, new_policy):
        self.policy = new_policy

    @abstractmethod
    def collect_experience(self):
        pass

    @abstractmethod
    def get_experience(self, trajectory_length) -> NestedTensor:
        pass


class BaseRolloutDriver(RolloutDriver):
    def __init__(self,
                 environment,
                 policy,
                 num_steps):
        super().__init__(environment, policy, num_steps)

    def collect_experience(self):
        observers = [self.replay_buffer.add_batch]
        driver = DynamicStepDriver(self.environment, self.policy, observers=observers, num_steps=self.num_steps)
        driver.run()

    def get_experience(self, trajectory_length=1):
        dataset = self.replay_buffer.as_dataset(sample_batch_size=self.num_steps, num_steps=trajectory_length)
        iterator = iter(dataset)
        experience, _ = next(iterator)
        return experience
