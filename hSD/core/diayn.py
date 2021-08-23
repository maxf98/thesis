from skill_discovery import SkillDiscovery

import tensorflow as tf
from tf_agents.environments.tf_environment import TFEnvironment

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import SkillModel
from core.modules.policy_learners import PolicyLearner


class DIAYN(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: BaseRolloutDriver,
                 skill_model: SkillModel,
                 policy_learner: PolicyLearner,
                 logger=None,
                 ):
        super(DIAYN, self).__init__(train_env, eval_env, rollout_driver, skill_model, policy_learner)
        self.logger = logger

    def get_discrim_trainset(self):
        """defines how the SD agent preprocesses the experience in the replay buffer to feed into the discriminator
        -- expects trajectories labelled with ground truth z in replay buffer
        --> turns them into suitable format for skill_discriminator"""

        dataset = self.rollout_driver.replay_buffer.as_dataset(single_deterministic_pass=True)  # for now we train the discriminator on all experience collected in the last epoch
        aug_obs = tf.stack(list(dataset.map(lambda x, _: x.observation)))
        s, z = self.split_observation(aug_obs)
        return s, z

    def get_rl_trainset(self, batch_size):
        """defines how reward-labelling and (maybe) data augmentation are performed by the agent before rl training"""
        dataset = self.rollout_driver.replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2)
        dataset_iter = iter(dataset)
        experience, _ = next(dataset_iter)
        reward_relabelled_experience = self.relabel_ir(experience)

        return reward_relabelled_experience

    def split_observation(self, aug_obs):
        s, z = tf.split(aug_obs, [self.skill_model.input_dim, self.skill_model.latent_dim], -1)
        return s, z

    def relabel_ir(self, batch):  # expects 2-time-step traj (as in SAC)
        obs_l, obs_r = tf.split(batch.observation, [1, 1], 1)  # for some reason this isn't working
        rl, rr = self.get_reward(obs_l), self.get_reward(obs_r)
        ir = tf.stack([rl, rr], axis=1)

        relabelled_batch = batch.replace(reward=ir)
        return relabelled_batch

    def get_reward(self, batch):  # expects single time-step batch (only appropriate for this version of DIAYN)
        batch = tf.squeeze(batch)
        s, z = self.split_observation(batch)
        log_probs = self.skill_model.log_probs(s, z)
        r = tf.subtract(log_probs, tf.math.log(1 / self.skill_model.latent_dim))
        return r

    def log_epoch(self, epoch, discrim_info, rl_info):
        if self.logger is not None:
            self.logger.log(epoch, discrim_info, rl_info, self.policy_learner.policy, self.skill_model, self.eval_env, self.skill_model.latent_dim)

    def save(self):
        if self.logger is not None:
            #self.logger.save_discrim(self.skill_discriminator)
            #self.logger.save_policy(self.policy_learner.agent.policy)
            #self.logger.save_stats()
            pass