import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from env.maze import maze_env
from env import point_environment, point_env_vis, skill_environment

from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments.tf_py_environment import TFPyEnvironment


from core.modules import utils


def hierarchy_vis(experiment_dir):
    # expects: layer_i, ..., policies, config file
    # we will parse environment from config file and visualise
    config = parse_config_file(experiment_dir)

    skill_lengths = (10, 10)

    envs = [get_env(config["env_name"])]
    #envs = [get_env("point_env")]
    base_env_obs_dim = envs[0].observation_spec().shape[0]
    skill_dim = int(config["skill_dim"])
    time_step_spec = ts.time_step_spec(utils.aug_obs_spec(envs[0].observation_spec(), base_env_obs_dim + skill_dim))

    policies = []
    skills = utils.discretize_continuous_space(-1, 1, 3, 2)

    num_layers = int(config["num_layers"])

    fig, axes = plt.subplots(1, num_layers + 1, figsize=(5, 5))

    for i in range(num_layers):
        print(f"visualising layer {i}")
        policy_dir = os.path.join(experiment_dir, f"policies/policy_{i}")
        layer_py_policy = SavedModelPyTFEagerPolicy(policy_dir, action_spec=envs[-1].action_spec(), time_step_spec=time_step_spec)
        layer_tf_policy = tf.compat.v2.saved_model.load(policy_dir)
        policies.append(layer_py_policy)

        tf_env = TFPyEnvironment(envs[-1])
        random_policy = RandomTFPolicy(time_step_spec, tf_env.action_spec())

        point_env_vis.skill_vis(axes[i], tf_env, layer_tf_policy, skills, 3, skill_lengths[i])
        axes[i].set(title=f"Layer {i}")

        skill_env = skill_environment.SkillEnv(envs[-1], layer_py_policy, skill_lengths[i])
        envs.append(skill_env)

    """
    tf_env = TFPyEnvironment(envs[-1])
    random_policy = RandomTFPolicy(time_step_spec, tf_env.action_spec())

    point_env_vis.skill_vis(axes[-1], tf_env, random_policy, skills, 3, 24)
    """
    plt.show()


def get_env(name): # copied from experiment launcher
    if name == "point_env":
        return point_environment.PointEnv(step_size=0.2, box_size=5)
    elif name.startswith("maze"):
        maze_type = name.split("_", 1)[1]
        return maze_env.MazeEnv(maze_type=maze_type)


def parse_config_file(dir):
    config_file = None
    for file in os.listdir(dir):
        if file.endswith(".gin"):
            config_file = os.path.join(dir, file)

    if config_file is not None:
        f = open(config_file, "r")
        param_dict = dict()
        for line in f:
            try:
                line = line.strip()
                line = line.replace(" ", "")
                line = line.replace("\"", "")
                line = line.split("=")
                lhs, rhs = line[0], line[1]
                param_name = lhs.split(".")[1]
                param_dict[param_name] = rhs
            except IndexError: # raised on blank lines
                continue

        f.close()

        return param_dict
    else:
        raise ValueError("config file not found in directory")


if __name__=="__main__":
    hierarchy_vis("../logs/diayn/hierarchy/anotha-test")