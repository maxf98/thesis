from abc import ABC, abstractmethod

from tqdm import tqdm
import time
from tf_agents.environments.tf_environment import TFEnvironment

from core.rollout_driver import RolloutDriver
from core.skill_discriminator import SkillDiscriminator
from core.policy_learner import PolicyLearner


class SkillDiscovery(ABC):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: RolloutDriver,
                 skill_discriminator: SkillDiscriminator,
                 policy_learner: PolicyLearner,
                 logger,
                 ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.rollout_driver = rollout_driver
        self.skill_discriminator = skill_discriminator
        self.policy_learner = policy_learner
        self.logger = logger

    @abstractmethod
    def preprocess_for_discriminator_training(self, batch):
        pass

    @abstractmethod
    def preprocess_for_rl_training(self, batch):
        pass

    def train(self,
              num_epochs,
              initial_collect_steps,
              collect_steps_per_epoch):

        for epoch in range(1, num_epochs + 1):
            tqdm.write(f"\nepoch {epoch}")
            # collect transitions from environment -- EXPLORE
            tqdm.write("EXPLORE")
            self.rollout_driver.collect_experience()
            experience = self.rollout_driver.get_experience()

            # train skill_discriminator on transitions -- DISCOVER
            time.sleep(0.5)
            tqdm.write("DISCOVER")
            x, y = self.preprocess_for_discriminator_training(experience)
            discriminator_training_info = self.skill_discriminator.train(x, y)

            # train rl_agent to optimize skills -- LEARN
            time.sleep(0.5)
            tqdm.write("LEARN")
            rl_train_batch = self.preprocess_for_rl_training(experience)
            rl_train_info = self.policy_learner.train(rl_train_batch)

            # log losses, times, and possibly visualise
            time.sleep(0.5)
            tqdm.write("logging")
            self.logger.log(epoch, discriminator_training_info, rl_train_info)



