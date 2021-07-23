from abc import ABC, abstractmethod

from tqdm import tqdm
import time
from tf_agents.environments.tf_environment import TFEnvironment

from core.rollout_drivers import RolloutDriver
from core.skill_discriminators import SkillDiscriminator
from core.policy_learners import PolicyLearner
from core.policies import SkillConditionedPolicy


class SkillDiscovery(ABC):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: RolloutDriver,
                 skill_discriminator: SkillDiscriminator,
                 policy_learner: PolicyLearner
                 ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.rollout_driver = rollout_driver
        self.skill_discriminator = skill_discriminator
        self.policy_learner = policy_learner

    @abstractmethod
    def get_discrim_trainset(self):
        """defines how the SD agent preprocesses the experience in the replay buffer to feed into the discriminator
        -- unfortunately need to process before adding to buffer and after, depending on how skill-labels
         of trajectory are induced"""
        pass

    @abstractmethod
    def get_rl_trainset(self, batch_size):
        """defines how reward-labelling and (maybe) data augmentation are performed by the agent before rl training"""
        pass

    @abstractmethod
    def log_epoch(self, epoch, discrim_info, rl_info):
        pass

    @abstractmethod
    def save(self):
        pass

    def train(self,
              num_epochs,
              initial_collect_steps,
              collect_steps_per_epoch,
              batch_size,
              discrim_epochs,
              sac_train_steps_per_epoch,  # other hyperparameters specific to training? (and therefore not any constituent module)
              ):

        for epoch in range(1, num_epochs + 1):
            tqdm.write(f"\nepoch {epoch}")
            # collect transitions from environment -- EXPLORE
            tqdm.write("EXPLORE")
            self.rollout_driver.collect_experience(initial_collect_steps if epoch == 1 else collect_steps_per_epoch)

            # train skill_discriminator on transitions -- DISCOVER
            time.sleep(0.5)
            tqdm.write("DISCOVER")
            x, y = self.get_discrim_trainset()
            discrim_history = self.skill_discriminator.train(x, y, batch_size=batch_size, epochs=discrim_epochs)
            discrim_train_stats = {'loss': discrim_history.history['loss'], 'accuracy': discrim_history.history['accuracy']}

            # train rl_agent to optimize skills -- LEARN
            time.sleep(0.5)
            tqdm.write("LEARN")
            sac_train_stats = {'loss': []}
            for _ in tqdm(range(sac_train_steps_per_epoch)):
                experience = self.get_rl_trainset(batch_size=batch_size)
                l = self.policy_learner.train(experience)
                sac_train_stats['loss'].append(l)

            self.rollout_driver.policy = self.policy_learner.agent.policy

            # log losses, times, and possibly visualise
            time.sleep(0.5)
            tqdm.write("logging")
            self.log_epoch(epoch, discrim_train_stats, sac_train_stats)

        self.save()


