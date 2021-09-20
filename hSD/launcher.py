import gin
import os
import shutil

import tensorflow as tf
import tensorflow_probability

from env import point_environment, skill_environment
from env.maze import maze_env
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, suite_gym
from tf_agents.policies import py_tf_eager_policy

from core.modules.rollout_drivers import BaseRolloutDriver
from core.modules.skill_models import BaseSkillModel, SkillDynamics
from core.modules.policy_learners import SACLearner
from core.dads import DADS
from core.diayn import DIAYN, ContDIAYN
from core.modules.logger import Logger
from core.modules import utils

tfd = tensorflow_probability.distributions



@gin.configurable
def hierarchical_skill_discovery(num_layers: int, skill_lengths, log_dir, config_path, training=False):
    """if num_layers == 1 we are simply performing skill discovery (no hierarchy)"""
    envs = [get_base_env()]
    agents = []

    create_log_dir(log_dir, config_path)

    for i in range(num_layers):
        print("–––––––––––––––––––––––––––––––––––")
        print(f"TRAINING LAYER {i}")
        print("–––––––––––––––––––––––––––––––––––")

        train_env, eval_env = envs[-1], envs[-1]
        skill_length = skill_lengths[i]

        # perform skill discovery on the current (wrapped) environment
        agent = initialise_skill_discovery_agent(tf_py_environment.TFPyEnvironment(train_env),
                                                 tf_py_environment.TFPyEnvironment(eval_env),
                                                 skill_length=skill_length,
                                                 log_dir=get_layer_log_dir(log_dir, i))

        policy, skill_model = agent.train()

        # embed learned skills in environment with SkillEnv
        py_policy = py_tf_eager_policy.PyTFEagerPolicy(policy)  # convert tf policy to py policy for skill wrapper
        envs.append(skill_environment.SkillEnv(train_env, py_policy, skill_length))

        agents.append(agent)

    return envs, agents


@gin.configurable
def get_base_env(env_name, point_env_step_size=0.1, point_env_box_size=1.) -> py_environment.PyEnvironment:
    """parses environment from name (string) and environment-specific hyperparameters"""
    if env_name == "point_env":
        return point_environment.PointEnv(step_size=point_env_step_size, box_size=point_env_box_size)
    elif env_name.startswith("maze"):
        maze_type = env_name.split("_", 1)[1]
        return maze_env.MazeEnv(maze_type=maze_type, action_range=point_env_step_size)
    elif env_name == "hopper":
        return suite_gym.load("Hopper-v2")
    elif env_name == "half-cheetah":
        return suite_gym.load("HalfCheetah-v2")
    elif env_name == "ant":
        return suite_gym.load("Ant-v2")


def create_log_dir(log_dir, config_path):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        shutil.copy(config_path, log_dir)
    else:
        print("directory exists already, you probably wanna handle that somehow...")


def get_layer_log_dir(log_dir, layer):
    return os.path.join(log_dir, str(layer))


@gin.configurable
def initialise_skill_discovery_agent(train_env, eval_env, skill_length, objective, skill_prior, skill_dim, log_dir):
    obs_spec, action_spec, time_step_spec = parse_env_specs(train_env, skill_dim, objective)

    skill_prior_distribution, vis_skill_set = parse_skill_prior(skill_dim, skill_prior)

    logger = Logger(log_dir,
                    vis_skill_set=vis_skill_set,
                    skill_length=skill_length)

    policy_learner = SACLearner(obs_spec,
                                action_spec,
                                time_step_spec)

    driver = BaseRolloutDriver(train_env,
                               policy_learner.policy,
                               skill_prior_distribution,
                               skill_length=skill_length)

    skill_model = init_skill_model(objective, skill_prior, train_env.observation_spec().shape[0], skill_dim)

    agent = init_skill_discovery(objective, skill_prior, train_env, eval_env, driver, skill_model, policy_learner,
                                 skill_dim, logger)

    logger.initialize_or_restore(agent)

    return agent


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
        return tfd.OneHotCategorical(logits=tf.ones(skill_dim), dtype=tf.float32), \
               tf.one_hot(list(range(skill_dim)), skill_dim)
    elif skill_prior == 'cont_uniform':
        vis_skill_set = utils.discretize_continuous_space(-1, 1, 3, skill_dim)
        # vis_skill_set = [[-1., 1.], [1., 1.], [-1., -1.], [1., -1.]]
        return tfd.Uniform(low=[-1.] * skill_dim, high=[1.] * skill_dim), vis_skill_set
    elif skill_prior == 'gaussian':
        vis_skill_set = utils.discretize_continuous_space(-1, 1, 3, skill_dim)
        return tfd.MultivariateNormalDiag(loc=[0.] * skill_dim, scale_diag=[1.] * skill_dim), vis_skill_set
    else:
        raise ValueError("invalid skill prior")


@gin.configurable
def init_skill_model(objective, skill_prior, obs_dim, skill_dim):
    if objective == "s->z":
        input_dim, output_dim = obs_dim, skill_dim
        return BaseSkillModel(input_dim,
                              output_dim,
                              skill_type=skill_prior)
    elif objective == "sz->s_p":
        input_dim, output_dim = obs_dim + skill_dim, obs_dim
        return BaseSkillModel(input_dim,
                              output_dim,
                              skill_type=skill_prior)
        # return SkillDynamics(input_dim, output_dim, fc_layer_params=hidden_dim, fix_variance=True)
    else:
        raise ValueError("invalid objective")


def init_skill_discovery(objective, skill_prior, train_env, eval_env, rollout_driver, skill_model, policy_learner,
                         skill_dim, logger):
    if objective == 's->z':
        if skill_prior == 'discrete_uniform':
            return DIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
        elif skill_prior == 'cont_uniform':
            return ContDIAYN(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)
    elif objective == 'sz->s_p':
        return DADS(train_env, eval_env, rollout_driver, skill_model, policy_learner, skill_dim, logger)

    else:
        raise ValueError("invalid objective")


if __name__ == '__main__':

    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    # we should still be able to run a plain skill discovery experiment...
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)

        hierarchical_skill_discovery(config_path=config_path)
