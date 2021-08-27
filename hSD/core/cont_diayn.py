from core.diayn import DIAYN

import tensorflow as tf
import numpy as np


class ContDIAYN(DIAYN):
    def get_reward(self, batch):
        """perform DADS-like marginalisation of sampled skills in a continuous space"""
        N = 100
        batch = tf.squeeze(batch)
        s, z = self.split_observation(batch)
        logp = self.skill_model.log_prob(s, z)

        alts = tf.concat([s] * N, axis=0)
        altz = self.rollout_driver.skill_prior.sample(alts.shape[0])
        logp_altz = tf.split(self.skill_model.log_prob(alts, altz), N, axis=0)

        intrinsic_reward = np.log(N + 1) - np.log(
            1 + np.exp(np.clip(logp_altz - tf.reshape(logp, (1, -1)), -50, 50)).sum(axis=0))

        return intrinsic_reward