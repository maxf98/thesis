from abc import ABC, abstractmethod

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from skill_discriminator import SkillDiscriminator

""" ADD ALL Skill Discovery SPECIFIC PARAMS HERE, """
num_epochs = 50
initial_collect_steps = 5000
collect_steps_per_epoch = 1000  # turn into collect_episodes ?
dynamics_train_steps_per_epoch = 32
sac_train_steps_per_epoch = 32

""" provide some demonstrative implementations of the abstract methods (?) """

class SkillDiscovery(ABC):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_discriminator: SkillDiscriminator,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 max_skill_length,
                 ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.max_skill_length = max_skill_length
        self.skill_discriminator = skill_discriminator
        self.rl_agent = rl_agent
        self.replay_buffer = replay_buffer

    @abstractmethod
    def _collect_env(self, steps):
        pass


    @abstractmethod
    def _train_discriminator(self, steps):
        pass

    @abstractmethod
    def _train_agent(self, steps):
        pass

    @abstractmethod
    def _log_epoch(self, epoch):
        pass

    def train(self):
        for epoch in range(1, num_epochs + 1):
            print("epoch {}".format(epoch))
            # collect transitions from environment -- EXPLORE
            self._collect_env(initial_collect_steps if epoch == 1 else collect_steps_per_epoch)

            # train skill_discriminator on transitions -- DISCOVER
            self._train_discriminator(dynamics_train_steps_per_epoch)

            # train rl_agent to optimize skills -- LEARN
            self._train_agent(sac_train_steps_per_epoch)

            # log losses, times, and possibly visualise
            self._log_epoch(epoch)
