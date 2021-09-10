import gin
import os

import tensorflow_probability

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import BaseSkillModel, SkillDynamics
from core.modules.policy_learners import SACLearner
from core.dads import DADS
from core.diayn import DIAYN, ContDIAYN
from core.modules.logger import Logger
from core.modules import utils

import tensorflow as tf

tfd = tensorflow_probability.distributions


@gin.configurable
def perform_skill_discovery(train_env, eval_env, skill_length, objective, skill_prior, skill_dim, log_dir):
    obs_spec, action_spec, time_step_spec = parse_env_specs(train_env, skill_dim, objective)

    logger = init_logger(log_dir=log_dir)

    skill_prior_distribution, vis_skill_set = parse_skill_prior(skill_dim, skill_prior)
    logger.vis_skill_set = vis_skill_set

    policy_learner = init_policy_learner(obs_spec, action_spec, time_step_spec)
    driver = init_rollout_driver(train_env, policy_learner.agent.collect_policy, skill_prior_distribution, skill_length=skill_length)
    skill_model = init_skill_model(objective, skill_prior, train_env.observation_spec().shape[0], skill_dim)

    # if there is a checkpointer already, this should restore the experiment state...
    # checkpointer expected in layer_log_dir/checkpointer
    # in this case, there should also be skill_model weights that we can copy somehow, optimiser state will be omitted
    # this, we will still need to figure out...
    """
    logger.initialise_checkpointer(agent=policy_learner.agent, replay_buffer=driver.replay_buffer,
                                   train_step=tf.compat.v1.train.get_global_step(),
                                   skill_model=skill_model)
    """
    agent = init_skill_discovery(objective, skill_prior, train_env, eval_env, driver, skill_model, policy_learner, skill_dim, logger)

    return train_skill_discovery(agent)


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
        vis_skill_set = utils.discretize_continuous_space(-1, 1, 3, skill_dim)
        #vis_skill_set = [[-1., 1.], [1., 1.], [-1., -1.], [1., -1.]]
        return tfd.Uniform(low=[-1.] * skill_dim, high=[1.] * skill_dim), vis_skill_set
    elif skill_prior == 'gaussian':
        vis_skill_set = utils.discretize_continuous_space(-1, 1, 3, skill_dim)
        return tfd.MultivariateNormalDiag(loc=[0.] * skill_dim, scale_diag=[1.] * skill_dim), vis_skill_set
    else:
        raise ValueError("invalid skill prior")


@gin.configurable
def init_rollout_driver(env, policy, skill_prior, state_norm=False, buffer_size=5000, skill_length=30, episode_length=30):
    return BaseRolloutDriver(env,
                             policy,
                             skill_prior,
                             buffer_size=buffer_size,
                             skill_length=skill_length,
                             episode_length=episode_length,
                             state_norm=state_norm)


@gin.configurable
def init_skill_model(objective, skill_prior, obs_dim, skill_dim, hidden_dim=(128, 128), fix_variance=False):
    if objective == "s->z":
        input_dim, output_dim = obs_dim, skill_dim
        return BaseSkillModel(input_dim,
                              output_dim,
                              skill_type=skill_prior,
                              fc_layer_params=hidden_dim,
                              fix_variance=fix_variance)
    elif objective == "sz->s_p":
        input_dim, output_dim = obs_dim + skill_dim, obs_dim
        return BaseSkillModel(input_dim,
                              output_dim,
                              skill_type=skill_prior,
                              fc_layer_params=hidden_dim,
                              fix_variance=fix_variance)
        #return SkillDynamics(input_dim, output_dim, fc_layer_params=hidden_dim, fix_variance=True)
    else:
        raise ValueError("invalid objective")


@gin.configurable
def init_policy_learner(obs_spec, action_spec, time_step_spec, rl_alg='SAC', fc_layer_params=(128, 128), target_entropy=None, reward_scale_factor=10.0, alpha_loss_weight=1.0):
    """enable other RL algorithms than SAC, or maybe just using a policy"""
    if rl_alg == 'SAC':
        return SACLearner(obs_spec,
                          action_spec,
                          time_step_spec,
                          network_fc_params=fc_layer_params,
                          target_entropy=target_entropy,
                          reward_scale_factor=reward_scale_factor,
                          alpha_loss_weight=alpha_loss_weight)


@gin.configurable
def init_logger(log_dir='.', skill_length=30, create_fig_interval=5, vis_skill_set=None, num_samples_per_skill=5):
    return Logger(log_dir,
                  create_fig_interval=create_fig_interval,
                  vis_skill_set=vis_skill_set,
                  skill_length=skill_length,
                  num_samples_per_skill=num_samples_per_skill)


def init_skill_discovery(objective, skill_prior, train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger):
    if objective == 's->z':
        if skill_prior == 'discrete_uniform':
            return DIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
        elif skill_prior == 'cont_uniform':
            return ContDIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)

    elif objective == 'sz->s_p':
        return DADS(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
    
    else:
        raise ValueError("invalid objective")


@gin.configurable
def train_skill_discovery(sd_agent, num_epochs=100, initial_collect_steps=5000, collect_steps_per_epoch=2000, train_batch_size=32, skill_model_train_steps=4, policy_learner_train_steps=128):
    return sd_agent.train(num_epochs,
                          initial_collect_steps,
                          collect_steps_per_epoch,
                          train_batch_size,
                          skill_model_train_steps,
                          policy_learner_train_steps)


if __name__ == '__main__':
    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    # we should still be able to run a plain skill discovery experiment... this isn't gonna work though, because we need to parse the environment...
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)

        perform_skill_discovery()
