import gin
from core.policies import SkillConditionedPolicy
from core.rollout_drivers import BaseRolloutDriver
from core.skill_discriminators import UniformCategoricalDiscriminator
from core.policy_learners import SACLearner
from env.point_environment import PointEnv
from core.diayn.diayn import DIAYN
from core.logger import Logger
from core import utils

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy


import tensorflow as tf
import tensorflow_probability as tfp
"""
launches DIAYN experiments -- want to subsume this into general launcher class
"""


def run_experiment():
    latent_dim = 4

    train_env = TFPyEnvironment(PointEnv())
    eval_env = TFPyEnvironment(PointEnv())

    obs_spec, action_spec, time_step_spec = train_env.observation_spec(), train_env.action_spec(), train_env.time_step_spec()
    obs_dim = obs_spec.shape.as_list()[0]
    aug_obs_spec = utils.aug_obs_spec(obs_spec, obs_dim + latent_dim)
    aug_ts_spec = utils.aug_time_step_spec(time_step_spec, obs_dim + latent_dim)

    skill_prior = tfp.distributions.OneHotCategorical(logits=tf.ones(latent_dim))

    initial_collect_policy = SkillConditionedPolicy(RandomTFPolicy(time_step_spec, action_spec), skill_prior)
    policy_learner = SACLearner(aug_obs_spec, action_spec, aug_ts_spec)

    driver = BaseRolloutDriver(train_env, initial_collect_policy, buffer_size=10000, skill_length=30, episode_length=30)
    discriminator = UniformCategoricalDiscriminator(train_env.observation_spec().numpy(), hidden_dim=(32, 32), latent_dim=latent_dim)
    policy_learner = SACLearner(aug_obs_spec, action_spec, aug_ts_spec)
    logger = Logger("../../logs", create_fig_interval=1)

    diayn_agent = DIAYN(train_env, eval_env, driver, discriminator, policy_learner, logger)

    diayn_agent.train(num_epochs=5, initial_collect_steps=2000, collect_steps_per_epoch=1000)


if __name__ == '__main__':
    run_experiment()
