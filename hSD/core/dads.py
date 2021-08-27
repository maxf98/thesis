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
                 skill_dim,
                 logger=None,
                 prior_samples=100,
                 ):
        super(DADS, self).__init__(train_env, eval_env, rollout_driver, skill_model, policy_learner)
        self.logger = logger

        self._skill_dim = skill_dim
        self._prior_samples = prior_samples

    def train_skill_model(self, batch_size, train_steps):
        # for now we train the discriminator on all experience in the replay buffer (rather than sampling from it repeatedly)
        dataset = self.rollout_driver.replay_buffer.as_dataset(num_steps=2, single_deterministic_pass=True)
        aug_obs = tf.stack(list(dataset.map(lambda x, _: x.observation)))
        s, z, ds_p = self.process_batch(aug_obs)
        history = self.skill_model.train(tf.concat([s, z], axis=-1), ds_p, batch_size=batch_size, epochs=train_steps)

        return {'loss': history.history['loss'], 'accuracy': history.history['accuracy']}

    def train_policy(self, batch_size, train_steps):
        dataset = self.rollout_driver.replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2)
        dataset_iter = iter(dataset)

        sac_stats = {'loss': [], 'reward': []}

        for _ in range(train_steps):
            experience, _ = next(dataset_iter)
            s, z, sp = self.process_batch(experience.observation)
            new_reward, _ = self.compute_dads_reward(s, z, sp)
            experience._replace(reward=tf.concat([np.expand_dims(new_reward, axis=1), experience.reward[:, 1:]], axis=1))
            iter_stats = self.policy_learner.train(experience)
            sac_stats['loss'].append(iter_stats)
            sac_stats['reward'].append(tf.reduce_mean(new_reward))

        return sac_stats

    def process_batch(self, batch):
        s = batch[:, 0, :self._skill_dim]
        z = batch[:, 0, -self._skill_dim:]
        s_p = batch[:, 1, :self._skill_dim]
        ds_p = s_p - s
        return s, z, ds_p

    def split_observation(self, aug_obs):
        s, z = tf.split(aug_obs, [-1, self._skill_dim], -1)
        return s, z

    def compute_dads_reward(self, input_obs, cur_skill, target_obs):
        num_reps = self._prior_samples if self._prior_samples > 0 else self._skill_dim - 1
        input_obs_altz = np.concatenate([input_obs] * num_reps, axis=0)
        target_obs -= input_obs
        target_obs_altz = np.concatenate([target_obs] * num_reps, axis=0)

        """ check this as well... am I sampling enough skills? I think so tbh..."""
        alt_skill = self.rollout_driver.skill_prior.sample(input_obs_altz.shape[0])

        logp = self.skill_model.log_prob(tf.concat([input_obs, cur_skill], axis=-1), target_obs)

        logp_altz = self.skill_model.log_prob(tf.concat([input_obs_altz, alt_skill], axis=-1), target_obs_altz)
        logp_altz = np.array(np.array_split(logp_altz, num_reps))

        # final DADS reward
        """this computation does not make sense to me... logp_altz - logp?
        I guess the sum does it, but I think it should come before the subtraction
        debug this..."""
        intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(np.clip(logp_altz - tf.reshape(logp, (1, -1)), -50, 50)).sum(axis=0))

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