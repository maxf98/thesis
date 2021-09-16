import gin
import os

import tensorflow_probability

import skill_discovery_launcher
import experiment_logger

from env import point_environment, skill_environment
from env.maze import maze_env
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, suite_gym

from tf_agents.policies import py_tf_eager_policy

import tensorflow as tf

tfd = tensorflow_probability.distributions



@gin.configurable
def hierarchical_skill_discovery(num_layers: int, skill_lengths, config_path):
    """if num_layers == 1 we are simply performing skill discovery (no hierarchy)"""
    envs = [get_base_env()]
    policies, skill_models = [], []

    logger = init_experiment_logger(config_path=config_path)

    for i in range(num_layers):
        print("–––––––––––––––––––––––––––––––––––")
        print(f"TRAINING LAYER {i}")
        print("–––––––––––––––––––––––––––––––––––")

        train_env, eval_env = envs[-1], envs[-1]
        skill_length = skill_lengths[i]

        # perform skill discovery on the current (wrapped) environment
        agent = skill_discovery_launcher.initialise_skill_discovery_agent(tf_py_environment.TFPyEnvironment(train_env),
                                                      tf_py_environment.TFPyEnvironment(eval_env),
                                                      skill_length=skill_length,
                                                      log_dir=logger.get_layer_log_dir(i))

        policy, skill_model = agent.train()

        # embed learned skills in environment with SkillEnv
        py_policy = py_tf_eager_policy.PyTFEagerPolicy(policy)  # convert tf policy to py policy for skill wrapper
        envs.append(skill_environment.SkillEnv(train_env, py_policy, skill_length))

        #keep cached for now, not really sure what for exactly...
        policies.append(policy)
        skill_models.append(skill_model)

    return policies, skill_models


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


@gin.configurable
def init_experiment_logger(log_dir, config_path=None):
    """should restore experiment state if possible, and handle saving and restoring policies and skill models"""
    return experiment_logger.ExperimentLogger(log_dir, config_path)


if __name__ == '__main__':
    config_root_dir = "configs/run-configs"
    configs = os.listdir(config_root_dir)
    print(configs)
    # we should still be able to run a plain skill discovery experiment...
    for config in configs:
        config_path = os.path.join(config_root_dir, config)
        gin.parse_config_file(config_path)

        hierarchical_skill_discovery(config_path=config_path)
