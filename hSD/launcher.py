import gin
import os

import tensorflow_probability

from core.policies import SkillConditionedPolicy
from core.rollout_drivers import BaseRolloutDriver
from core.skill_discriminators import UniformCategoricalDiscriminator, GaussianDiscriminator
from core.policy_learners import SACLearner
from env.point_environment import PointEnv
from core.diayn.diayn import DIAYN
from core.logger import Logger
from core import utils

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
"""
launches DIAYN experiments -- want to subsume this into general launcher class
"""


@gin.configurable
def run_experiment(latent_dim=4, config_path=None):
    train_env = TFPyEnvironment(PointEnv())
    eval_env = TFPyEnvironment(PointEnv())

    obs_spec, action_spec, time_step_spec = train_env.observation_spec(), train_env.action_spec(), train_env.time_step_spec()
    obs_dim = obs_spec.shape.as_list()[0]
    aug_obs_spec = utils.aug_obs_spec(obs_spec, obs_dim + latent_dim)
    aug_ts_spec = utils.aug_time_step_spec(time_step_spec, obs_dim + latent_dim)

    skill_prior, vis_skill_set = parse_skill_prior(latent_dim)

    policy_learner = init_policy_learner(aug_obs_spec, action_spec, aug_ts_spec)
    driver = init_rollout_driver(train_env, policy_learner.agent.collect_policy, skill_prior)
    discriminator = init_discriminator(train_env.observation_spec().shape[0], latent_dim)
    logger = init_logger(config_path=config_path, vis_skill_set=vis_skill_set)

    diayn_agent = DIAYN(train_env, eval_env, driver, discriminator, policy_learner, logger)

    train_skill_discovery(diayn_agent)


@gin.configurable
def parse_skill_prior(latent_dim, prior='UniformCategorical'):
    if prior == 'UniformCategorical':
        return tfd.OneHotCategorical(logits=tf.ones(latent_dim), dtype=tf.float32), tf.one_hot(list(range(latent_dim)), latent_dim)
    elif prior == 'UniformContinuous':
        vis_skill_set = discretize_continuous_space(-1, 1, 3)
        return tfd.Uniform(low=[-1.] * latent_dim, high=[1.] * latent_dim), vis_skill_set
    else:
        print("Invalid init")


def discretize_continuous_space(min, max, num_points):
    step = (max-min) / num_points
    return [[min + step * x, min + step * y] for x in range(num_points) for y in range(num_points)]


@gin.configurable
def init_rollout_driver(env, policy, skill_prior, buffer_size=5000, skill_length=30, episode_length=30):
    return BaseRolloutDriver(env, policy, skill_prior, buffer_size=buffer_size,
                             skill_length=skill_length, episode_length=episode_length)


@gin.configurable
def init_discriminator(input_dim, latent_dim, hidden_dim=(128, 128), outputs='OneHot'):
    if outputs == 'OneHot':
        return UniformCategoricalDiscriminator(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    elif outputs == 'MixtureOfGaussians':
        return GaussianDiscriminator(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    else:
        print("Invalid init")


@gin.configurable
def init_policy_learner(obs_spec, action_spec, time_step_spec, rl_alg='SAC', fc_layer_params=(128, 128), entropy_regularized=False):
    if rl_alg == 'SAC':
        return SACLearner(obs_spec, action_spec, time_step_spec, network_fc_params=fc_layer_params, entropy_regularized=entropy_regularized)


@gin.configurable
def init_logger(create_logs=True, log_dir='.', create_fig_interval=5, config_path=None, vis_skill_set=None):
    if create_logs:
        return Logger(log_dir, create_fig_interval=create_fig_interval, config_path=config_path, vis_skill_set=vis_skill_set)
    return None


@gin.configurable
def train_skill_discovery(sd_agent, num_epochs=100, initial_collect_steps=5000, collect_steps_per_epoch=2000, train_batch_size=32, discrim_epochs=4, sac_train_steps=128):
    sd_agent.train(num_epochs, initial_collect_steps, collect_steps_per_epoch, train_batch_size, discrim_epochs, sac_train_steps)


if __name__ == '__main__':
    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)
        run_experiment(config_path=config_path)
