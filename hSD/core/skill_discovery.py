from abc import ABC, abstractmethod
import gin

from tqdm import tqdm
import gtimer as gt
import tensorflow as tf
from tf_agents.environments.tf_environment import TFEnvironment

from core.modules.rollout_drivers import RolloutDriver
from core.modules.skill_models import SkillModel
from core.modules.policy_learners import PolicyLearner


class SkillDiscovery(ABC):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: RolloutDriver,
                 skill_model: SkillModel,
                 policy_learner: PolicyLearner
                 ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.rollout_driver = rollout_driver
        self.skill_model = skill_model
        self.policy_learner = policy_learner

    @abstractmethod
    def train_skill_model(self, batch_size, train_steps):
        """defines how the SD agent preprocesses the experience in the replay buffer to feed into the discriminator
        -- unfortunately need to process before adding to buffer and after, depending on how skill-labels
         of trajectory are induced"""
        pass

    @abstractmethod
    def train_policy(self, batch_size, train_steps):
        """defines how reward-labelling and (maybe) data augmentation are performed by the agent before rl training"""
        pass

    @abstractmethod
    def log_epoch(self, epoch, skill_model_info, rl_info, timer_info):
        pass

    @gin.configurable
    def train(self,
              num_epochs,
              initial_collect_steps,
              collect_steps_per_epoch,
              batch_size,
              skill_model_train_steps,
              policy_learner_train_steps
              ):

        for epoch in range(1, num_epochs + 1):
            print(f"\nepoch {epoch}")
            # collect transitions from environment -- EXPLORE
            print("Collect experience")
            self.rollout_driver.collect_experience(initial_collect_steps if epoch == 1 else collect_steps_per_epoch)
            gt.stamp("exploration", unique=False)

            # train skill_discriminator on transitions -- DISCOVER
            print("Train skill model")
            discrim_train_stats = self.train_skill_model(batch_size, skill_model_train_steps)
            gt.stamp("skill model training", unique=False)

            # train rl_agent to optimize skills -- LEARN
            print("Train policy")
            sac_train_stats = self.train_policy(batch_size, policy_learner_train_steps)
            gt.stamp("rl training", unique=False)

            # update exploration/collect policy
            self.rollout_driver.policy = self.policy_learner.collect_policy

            # log losses, times, and possibly visualise
            self.log_epoch(epoch, discrim_train_stats, sac_train_stats, gt.report())
            gt.stamp("logging", unique=False)

        print(gt.report())

        return self.policy_learner.policy, self.skill_model


