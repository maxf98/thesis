import gin
import os
import itertools

import tensorflow_probability

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import BaseSkillModel, SkillDynamics
from core.modules.policy_learners import SACLearner
from env.point_environment import PointEnv
from core.diayn import DIAYN
from core.dads import DADS
from core.modules.logger import Logger
from core.modules import utils

from tf_agents.environments.tf_py_environment import TFPyEnvironment

import tensorflow as tf

tfd = tensorflow_probability.distributions


@gin.configurable
def run_experiment(objective, skill_prior, skill_dim, config_path=None):
    train_env = TFPyEnvironment(PointEnv())
    eval_env = TFPyEnvironment(PointEnv())

    obs_spec, action_spec, time_step_spec = parse_env_specs(train_env, skill_dim, objective)

    skill_prior_distribution, vis_skill_set = parse_skill_prior(skill_dim, skill_prior)

    policy_learner = init_policy_learner(obs_spec, action_spec, time_step_spec)
    driver = init_rollout_driver(train_env, policy_learner.agent.collect_policy, skill_prior_distribution)
    skill_model = init_skill_model(objective, skill_prior, train_env.observation_spec().shape[0], skill_dim)
    logger = init_logger(config_path=config_path, vis_skill_set=vis_skill_set)

    agent = init_skill_discovery(objective, train_env, eval_env, driver, skill_model, policy_learner, skill_dim, logger)

    train_skill_discovery(agent)


def parse_env_specs(env, skill_dim, objective):
    obs_spec, action_spec, time_step_spec = env.observation_spec(), env.action_spec(), env.time_step_spec()
    obs_dim = obs_spec.shape.as_list()[0]

    if objective == 's->z' or objective == 'sz->s_p':  # diayn, dads
        obs_spec = utils.aug_obs_spec(obs_spec, obs_dim + skill_dim)
        time_step_spec = utils.aug_time_step_spec(time_step_spec, obs_dim + skill_dim)
    else:
        raise ValueError("invalid objective")
    return obs_spec, action_spec, time_step_spec


@gin.configurable
def parse_skill_prior(skill_dim, skill_prior):
    if skill_prior == 'discrete_uniform':
        return tfd.OneHotCategorical(logits=tf.ones(skill_dim), dtype=tf.float32), tf.one_hot(list(range(skill_dim)), skill_dim)
    elif skill_prior == 'cont_uniform':
        vis_skill_set = discretize_continuous_space(-1, 1, 3, skill_dim)
        return tfd.Uniform(low=[-1.] * skill_dim, high=[1.] * skill_dim), vis_skill_set
    elif skill_prior == 'gaussian':
        vis_skill_set = discretize_continuous_space(-1, 1, 3, skill_dim)
        return tfd.MultivariateNormalDiag(loc=[0.] * skill_dim, scale_diag=[1.] * skill_dim), vis_skill_set
    else:
        raise ValueError("invalid skill prior")


def discretize_continuous_space(min, max, num_points, dim):
    step = (max-min) / num_points
    skill_axes = [[min + step * x for x in range(num_points + 1)]] * dim
    skills = [skill for skill in itertools.product(*skill_axes)]
    return skills


@gin.configurable
def init_rollout_driver(env, policy, skill_prior, buffer_size=5000, skill_length=30, episode_length=30):
    return BaseRolloutDriver(env, policy, skill_prior, buffer_size=buffer_size,
                             skill_length=skill_length, episode_length=episode_length)


@gin.configurable
def init_skill_model(objective, skill_prior, obs_dim, skill_dim, hidden_dim=(128, 128), fix_variance=False):
    if objective == "s->z":
        input_dim, output_dim = obs_dim, skill_dim
        return BaseSkillModel(input_dim, output_dim, skill_type=skill_prior, fc_layer_params=hidden_dim,
                              fix_variance=fix_variance)
    elif objective == "sz->s_p":
        input_dim, output_dim = obs_dim + skill_dim, obs_dim
        return SkillDynamics(input_dim, output_dim, fix_variance=True)
    else:
        raise ValueError("invalid objective")



@gin.configurable
def init_policy_learner(obs_spec, action_spec, time_step_spec, rl_alg='SAC', fc_layer_params=(128, 128), target_entropy=None, reward_scale_factor=10.0):
    """enable other RL algorithms than SAC"""
    if rl_alg == 'SAC':
        return SACLearner(obs_spec, action_spec, time_step_spec, network_fc_params=fc_layer_params, target_entropy=target_entropy, reward_scale_factor=reward_scale_factor)


@gin.configurable
def init_logger(create_logs=True, log_dir='.', create_fig_interval=5, config_path=None, vis_skill_set=None, skill_length=30, num_samples_per_skill=5):
    if create_logs:
        return Logger(log_dir, create_fig_interval=create_fig_interval, config_path=config_path, vis_skill_set=vis_skill_set, skill_length=skill_length, num_samples_per_skill=num_samples_per_skill)
    return None


def init_skill_discovery(objective, train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger):
    if objective == 's->z':
        return DIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
    elif objective == 'sz->s_p':
        return DADS(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
    else:
        raise ValueError("invalid objective")


@gin.configurable
def train_skill_discovery(sd_agent, num_epochs=100, initial_collect_steps=5000, collect_steps_per_epoch=2000, train_batch_size=32, skill_model_train_steps=4, policy_learner_train_steps=128):
    sd_agent.train(num_epochs, initial_collect_steps, collect_steps_per_epoch, train_batch_size, skill_model_train_steps, policy_learner_train_steps)


if __name__ == '__main__':
    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)
        run_experiment(config_path=config_path)
