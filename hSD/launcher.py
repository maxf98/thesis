import gin
import os

import tensorflow_probability

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import UniformCategoricalSkillModel, GaussianSkillModel
from core.modules.policy_learners import SACLearner
from env.point_environment import PointEnv
from core.diayn import DIAYN
from core.dads import DADS
from core.modules.logger import Logger
from core.modules import utils

from tf_agents.environments.tf_py_environment import TFPyEnvironment

import tensorflow as tf

tfd = tensorflow_probability.distributions
"""
launches DIAYN experiments -- want to subsume this into general launcher class
"""


@gin.configurable
def run_experiment(objective="s->z", skill_prior='UniformCategorical', skill_dim=4, config_path=None):
    train_env = TFPyEnvironment(PointEnv())
    eval_env = TFPyEnvironment(PointEnv())

    obs_spec, action_spec, time_step_spec = parse_objective(train_env, skill_dim, objective)

    skill_prior_distribution, vis_skill_set = parse_skill_prior(skill_dim, skill_prior)

    policy_learner = init_policy_learner(obs_spec, action_spec, time_step_spec)
    driver = init_rollout_driver(train_env, policy_learner.agent.collect_policy, skill_prior_distribution)
    skill_model = init_skill_model(obs_spec.shape[0], train_env.observation_spec().shape[0], skill_prior=skill_prior)
    logger = init_logger(config_path=config_path, vis_skill_set=vis_skill_set)

    agent = init_skill_discovery(objective, train_env, eval_env, driver, skill_model, policy_learner, logger)

    train_skill_discovery(agent)


def parse_objective(env, skill_dim, objective):
    obs_spec, action_spec, time_step_spec = env.observation_spec(), env.action_spec(), env.time_step_spec()
    obs_dim = obs_spec.shape.as_list()[0]

    if objective == 's->z' or objective == 'sz->s_p':  # diayn, dads
        obs_spec = utils.aug_obs_spec(obs_spec, obs_dim + skill_dim)
        time_step_spec = utils.aug_time_step_spec(time_step_spec, obs_dim + skill_dim)
    else:
        print("objective undefined, this should actually be an error")
    return obs_spec, action_spec, time_step_spec


@gin.configurable
def parse_skill_prior(latent_dim, skill_prior):
    if skill_prior == 'UniformCategorical':
        return tfd.OneHotCategorical(logits=tf.ones(latent_dim), dtype=tf.float32), tf.one_hot(list(range(latent_dim)), latent_dim)
    elif skill_prior == 'UniformContinuous':
        vis_skill_set = discretize_continuous_space(-1, 1, 3)
        return tfd.Uniform(low=[-1.] * latent_dim, high=[1.] * latent_dim), vis_skill_set
    elif skill_prior == 'Gaussian':
        print("not implemented yet... (one line of code)")
    else:
        print("invalid skill prior given, this should raise an error")


def discretize_continuous_space(min, max, num_points):
    step = (max-min) / num_points
    return [[min + step * x, min + step * y] for x in range(num_points) for y in range(num_points)]


@gin.configurable
def init_rollout_driver(env, policy, skill_prior, buffer_size=5000, skill_length=30, episode_length=30):
    return BaseRolloutDriver(env, policy, skill_prior, buffer_size=buffer_size,
                             skill_length=skill_length, episode_length=episode_length)


@gin.configurable
def init_skill_model(input_dim, latent_dim, hidden_dim=(128, 128), skill_prior='UniformCategorical', fix_variance=False):
    if skill_prior == 'UniformCategorical':
        return UniformCategoricalSkillModel(input_dim, hidden_dims=hidden_dim, latent_dim=latent_dim)
    elif skill_prior in ['UniformContinous', 'Gaussian']:
        return GaussianSkillModel(input_dim, hidden_dims=hidden_dim, latent_dim=latent_dim, fix_variance=fix_variance)
    else:
        print("invalid skill prior given, this should raise an error")


@gin.configurable
def init_policy_learner(obs_spec, action_spec, time_step_spec, rl_alg='SAC', fc_layer_params=(128, 128), target_entropy=None):
    if rl_alg == 'SAC':
        return SACLearner(obs_spec, action_spec, time_step_spec, network_fc_params=fc_layer_params, target_entropy=target_entropy)


@gin.configurable
def init_logger(create_logs=True, log_dir='.', create_fig_interval=5, config_path=None, vis_skill_set=None):
    if create_logs:
        return Logger(log_dir, create_fig_interval=create_fig_interval, config_path=config_path, vis_skill_set=vis_skill_set)
    return None


def init_skill_discovery(objective, train_env, eval_env, rollout_driver, skill_model, policy_learner, logger):
    if objective == 's->z':
        return DIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, logger)
    elif objective == 'sz->s_p':
        return DADS(train_env, eval_env, rollout_driver, skill_model, policy_learner, logger, skill_dim=2)
    else:
        print('objective undefined, we should not have gotten here')
        return None


@gin.configurable
def train_skill_discovery(sd_agent, num_epochs=100, initial_collect_steps=5000, collect_steps_per_epoch=2000, train_batch_size=32, skill_model_train_steps=4, policy_learner_train_steps=128):
    sd_agent.train(num_epochs, initial_collect_steps, collect_steps_per_epoch, train_batch_size, skill_model_train_steps, policy_learner_train_steps)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)
        run_experiment(config_path=config_path)
