import gin
import os
from core.modules import utils
import launcher
import numpy as np
import matplotlib.pyplot as plt
from scripts import point_env_vis


def load_trained_agent(config_path):
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)
    l0_agent = agents[0]

    skills = utils.discretize_continuous_space(-1, 1, 3, 2)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    point_env_vis.skill_vis(ax, l0_agent.eval_env, l0_agent.policy_learner.policy, skills, 3, skill_length=100)
    point_env_vis.config_subplot(ax, box_size=2.5)
    plt.show()



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
    config_path = "/Users/maxfest/RL/thesis/hSD/logs/diayn/thesis/hiercomp/flat/config.gin"
    load_trained_agent(config_path)
    #inspect_stats("../logs/diayn/thesis/hopper/0/stats")