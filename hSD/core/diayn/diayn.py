from skill_discovery import SkillDiscovery

from tqdm import tqdm
import time
from tf_agents.environments.tf_environment import TFEnvironment

from core.rollout_drivers import BaseRolloutDriver
from core.skill_discriminators import SkillDiscriminator
from core.policy_learners import PolicyLearner


class DIAYN(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: BaseRolloutDriver,
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

    def get_discrim_trainset(self):
        """defines how the SD agent preprocesses the experience in the replay buffer to feed into the discriminator
        -- expects trajectories labelled with ground truth z in replay buffer
        --> turns them into suitable format for skill_discriminator"""

        trajectories = self.rollout_driver.get_experience(num_steps=1)
        obs = trajectories.observation

    def get_rl_trainset(self):
        """defines how reward-labelling and (maybe) data augmentation are performed by the agent before rl training"""
        pass

    def log_epoch(self, epoch, discrim_info, rl_info):
        self.logger.log(epoch, discrim_info, rl_info)