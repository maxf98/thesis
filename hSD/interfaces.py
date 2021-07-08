from abc import ABC, abstractmethod
from tqdm import tqdm

from tf_agents.drivers.driver import Driver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import trajectory


class RolloutDriver(ABC):
    def __init__(self, environment: TFEnvironment, skill_length, episode_length, collect_steps):
        """
        Encapsulates all the logic for collecting experience from an environment, therefore a class that
        inherits from RolloutDriver defines its exploration policy... or nah... not really tbh
        I think the buffer should be part of SkillDiscovery...
        Parameters
        ----------
        environment
        skill_length
        episode_length
        collect_steps
        """
        self.environment = environment
        self.skill_length = skill_length
        self.episode_length = episode_length
        self.collect_steps = collect_steps

        self.exploration_data = []

    @abstractmethod
    def collect_experience(self) -> ReplayBuffer:
        """

        Returns
        -------
        returns a replay_buffer with collected experience
        """
        pass


class SkillDiscriminator(ABC):
    def __init__(self, input_dim, intermediate_dim, latent_dim, supervised=False):
        """
        builds model with specified architecture, either to perform unsupervised training with reconstruction,
        or DIAYN-style supervised training
        Parameters
        ----------
        input_dim
        intermediate_dim
        latent_dim
        """

