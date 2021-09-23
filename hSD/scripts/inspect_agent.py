import gin
import os
import gym
from core.modules import utils
import launcher
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.policies import py_tf_eager_policy
from tf_agents.environments import suite_gym
from tf_agents.specs import BoundedArraySpec


def load_agent(config_path):
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)
    agent = agents[0]
    base_env, policy, skill_model = agent.eval_env, agent.policy_learner.policy, agent.skill_model

    timestep = base_env.reset()
    z = [-0.5, 0.5]
    for _ in range(2000):
        base_env.render(mode='human')
        aug_ts = utils.aug_time_step(timestep, z)
        action_step = policy.action(aug_ts)
        timestep = base_env.step(action_step.action)


def inspect_stats(dir):
    ir = np.load(os.path.join(dir, "intrinsic_rewards.npy")).tolist()
    sl = np.load(os.path.join(dir, "policy_loss.npy")).tolist()
    dl= np.load(os.path.join(dir, "discrim_loss.npy")).tolist()
    da = np.load(os.path.join(dir, "discrim_acc.npy")).tolist()

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(range(len(da)), da, color='lightblue', linewidth=3)
    ax1.set(title='Skill model accuracy')

    ax2.plot(range(len(ir)), ir, color='green', linewidth=3)
    ax2.set(title='intrinsic reward')

    plt.show()


def run_saved_policy():
    latent_dim = 2
    env = suite_gym.load("HalfCheetah-v2")
    ts_spec = env.time_step_spec()
    aug_obs_spec = ts_spec.observation.replace(shape=(ts_spec.observation.shape[0] + latent_dim, ))
    time_step_spec = ts_spec._replace(observation=aug_obs_spec)
    action_spec = env.action_spec()

    policy_dir = "/home/max/RL/thesis/hSD/logs/halfcheetah/0/policies/policy_1000"
    pypolicy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(policy_dir, time_step_spec, action_spec)

    timestep = env.reset()
    z = [-1.0, 1.0]
    for _ in range(2000):
        env.render(mode='human')
        aug_ts = utils.aug_time_step(timestep, z)
        action_step = pypolicy.action(aug_ts)
        timestep = env.step(action_step.action)


if __name__=='__main__':
    config_path = "/home/max/RL/thesis/hSD/logs/hopper/config.gin"
    #load_agent(config_path)
    #inspect_stats("../logs/diayn/thesis/hopper/0/stats")

    run_saved_policy()

    #load_agent(config_path)
