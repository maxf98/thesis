import gin
import os
from core.modules import utils
import launcher
import numpy as np
import matplotlib.pyplot as plt


def load_agent(config_path):
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)
    agent = agents[0]
    base_env, policy, skill_model = agent.eval_env, agent.policy_learner.policy, agent.skill_model

    timestep = base_env.reset()
    z = [-0.5, 0.5]
    for _ in range(100):
        base_env.render()
        aug_ts = utils.aug_time_step(timestep, z)
        action_step = policy.action(aug_ts)
        base_env.step(action_step.action)


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


if __name__=='__main__':
    config_path = "~/RL/thesis/hSD/logs/diayn/thesis/hopper/config.gin"
    load_agent(config_path)
    #inspect_stats("../logs/diayn/thesis/hopper/0/stats")