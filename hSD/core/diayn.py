from core.skill_discovery import SkillDiscovery

import tensorflow as tf
import numpy as np
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
                 skill_dim,
                 logger=None,
                 ):
        super(DIAYN, self).__init__(train_env, eval_env, rollout_driver, skill_model, policy_learner)
        self.skill_dim = skill_dim
        self.logger = logger

    def train_skill_model(self, batch_size, train_steps):
        dataset = self.rollout_driver.replay_buffer.as_dataset(
            single_deterministic_pass=True)  # for now we train the discriminator on all experience collected in the last epoch
        aug_obs = tf.stack(list(dataset.map(lambda x, _: x.observation)))
        s, z = self.split_observation(aug_obs)
        history = self.skill_model.train(s, z, batch_size=batch_size, epochs=train_steps)
        return {'loss': history.history['loss'], 'accuracy': history.history['accuracy']}

    def train_policy(self, batch_size, train_steps):
        dataset = self.rollout_driver.replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2)
        dataset_iter = iter(dataset)

        sac_stats = {'loss': [], 'reward': []}

        for _ in range(train_steps):
            experience, _ = next(dataset_iter)
            reward_relabelled_experience = self.relabel_ir(experience)
            iter_stats = self.policy_learner.train(reward_relabelled_experience)
            sac_stats['loss'].append(iter_stats)
            sac_stats['reward'].append(tf.reduce_mean(reward_relabelled_experience.reward[:, 0]))

        return sac_stats

    def split_observation(self, aug_obs):
        s, z = tf.split(aug_obs, [-1, self.skill_dim], -1)
        return s, z

    def relabel_ir(self, batch):  # expects 2-time-step traj (as in SAC)
        obs_l, obs_r = tf.split(batch.observation, [1, 1], 1)
        rl, rr = self.get_reward(obs_l), self.get_reward(obs_r)  # not sure we actually need to compute reward for both time steps...
        ir = tf.stack([rl, rr], axis=1)

        relabelled_batch = batch.replace(reward=ir)
        return relabelled_batch

    def get_reward(self, batch):  # expects single time-step batch (only appropriate for this version of DIAYN)
        batch = tf.squeeze(batch)
        s, z = self.split_observation(batch)
        log_probs = self.skill_model.log_prob(s, z)
        r = tf.subtract(log_probs, tf.math.log(1 / self.skill_dim))
        return r

    def log_epoch(self, epoch, skill_stats, sac_stats):
        global_step = tf.compat.v1.train.get_global_step()
        if self.logger is not None:
            self.logger.log(epoch, global_step, skill_stats, sac_stats, self.policy_learner.policy, self.skill_model, self.eval_env)
            #self.logger.per_skill_collect_rollouts(epoch, self.policy_learner.collect_policy, self.eval_env)


class ContDIAYN(DIAYN):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 rollout_driver: BaseRolloutDriver,
                 skill_model: SkillModel,
                 policy_learner: PolicyLearner,
                 skill_dim,
                 logger=None,
                 num_prior_samples=100
                 ):
        super(DIAYN, self).__init__(train_env, eval_env, rollout_driver, skill_model, policy_learner)
        self.skill_dim = skill_dim
        self.logger = logger
        self.num_prior_samples = num_prior_samples

    def get_reward(self, batch):
        """perform DADS-like marginalisation of sampled skills in a continuous space"""
        N = self.num_prior_samples
        batch = tf.squeeze(batch)
        s, z = self.split_observation(batch)
        logp = self.skill_model.log_prob(s, z)

        alts = tf.concat([s] * N, axis=0)
        altz = self.rollout_driver.skill_prior.sample(alts.shape[0])
        logp_altz = tf.split(self.skill_model.log_prob(alts, altz), N, axis=0)

        intrinsic_reward = np.log(N + 1) - np.log(
            1 + np.exp(np.clip(logp_altz - tf.reshape(logp, (1, -1)), -50, 50)).sum(axis=0))

        return intrinsic_reward