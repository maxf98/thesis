from skill_discovery import SkillDiscovery

import tensorflow as tf
import numpy as np
from tf_agents.environments.tf_environment import TFEnvironment

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import SkillModel
from core.modules.policy_learners import PolicyLearner


class DADS(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: BaseRolloutDriver,
                 skill_model: SkillModel,
                 policy_learner: PolicyLearner,
                 logger=None,
                 skill_dim=4,
                 prior_samples=100,
                 ):
        super(DADS, self).__init__(train_env, eval_env, rollout_driver, skill_model, policy_learner)
        self.logger = logger

        self._skill_dim = skill_dim
        self._prior_samples = prior_samples

    def get_discrim_trainset(self):
        """defines how the SD agent preprocesses the experience in the replay buffer to feed into the discriminator
        -- expects trajectories labelled with ground truth z in replay buffer
        --> turns them into suitable format for skill_discriminator"""

        dataset = self.rollout_driver.replay_buffer.as_dataset(num_steps=2, single_deterministic_pass=True)  # for now we train the discriminator on all experience collected in the last epoch
        aug_obs = tf.stack(list(dataset.map(lambda x, _: x.observation)))
        s, z, ds_p = self.process_batch(aug_obs)
        return tf.concat([s, z], axis=-1), ds_p

    def get_rl_trainset(self, batch_size):
        """defines how reward-labelling and (maybe) data augmentation are performed by the agent before rl training"""
        dataset = self.rollout_driver.replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2)
        dataset_iter = iter(dataset)
        experience, _ = next(dataset_iter)
        s, z, sp = self.process_batch(experience.observation)
        new_reward, _ = self.compute_dads_reward(s, z, sp)
        experience._replace(
            reward=tf.concat(
                [np.expand_dims(new_reward, axis=1), experience.reward[:, 1:]],
                axis=1))

        return experience

    def process_batch(self, batch):
        s = batch[:, 0, :self._skill_dim]
        z = batch[:, 0, -self._skill_dim:]
        s_p = batch[:, 1, :self._skill_dim]
        ds_p = s_p - s
        return s, z, ds_p

    def split_observation(self, aug_obs):
        s, z = tf.split(aug_obs, [self.skill_model.input_dim, self.skill_model.latent_dim], -1)
        return s, z

    def compute_dads_reward(self, input_obs, cur_skill, target_obs):
        num_reps = self._prior_samples if self._prior_samples > 0 else self._skill_dim - 1
        input_obs_altz = np.concatenate([input_obs] * num_reps, axis=0)
        target_obs_altz = np.concatenate([target_obs] * num_reps, axis=0)

        alt_skill = self.rollout_driver.skill_prior.sample(input_obs_altz.shape[0])

        logp = self.skill_model.log_prob(tf.concat([input_obs, cur_skill], axis=-1), target_obs)

        # denominator may require more memory than that of a GPU, break computation
        logp_altz = self.skill_model.log_prob(tf.concat([input_obs_altz, alt_skill], axis=-1), target_obs_altz)

        logp_altz = np.array(np.array_split(logp_altz, num_reps))

        # final DADS reward
        intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(
            np.clip(logp_altz - tf.reshape(logp, (1, -1)), -50, 50)).sum(axis=0))

        return intrinsic_reward, {'logp': logp, 'logp_altz': logp_altz.flatten()}

    def log_epoch(self, epoch, discrim_info, rl_info):
        if self.logger is not None:
            self.logger.log(epoch, discrim_info, rl_info, self.policy_learner.policy, self.skill_model, self.eval_env)

    def save(self):
        if self.logger is not None:
            #self.logger.save_discrim(self.skill_discriminator)
            #self.logger.save_policy(self.policy_learner.agent.policy)
            #self.logger.save_stats()
            pass