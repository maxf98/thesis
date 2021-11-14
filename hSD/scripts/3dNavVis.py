import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from env import point_environment
from scripts import point_env_vis

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_py_policy import RandomPyPolicy

from core.modules import utils


def vis_random_rollouts():
    env = point_environment.PointEnv(step_size=0.1, box_size=1., dim=3)
    policy = RandomPyPolicy(env.time_step_spec(), env.action_spec())
    skills = [[0.]]*100

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    skill_vis(ax, env, policy, skills, 1, 20)

    plt.show()


def vis_saved_policy(policy_dir, cont=True, title=None, env=None, skill_length=25, step_size=0.1, box_size=1.):
    policy = tf.compat.v2.saved_model.load(policy_dir)
    if env is None:
        env = point_environment.PointEnv(step_size=step_size, box_size=box_size, dim=3)
    if cont:
        #skills = utils.discretize_continuous_space(-1, 1, 2, 2)
        skills = [[-1., -1.]]
    else:
        NUM_SKILLS = 8
        skills = tf.one_hot(list(range(NUM_SKILLS)), NUM_SKILLS)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    skill_vis(ax, env, policy, skills, 1, skill_length=skill_length, box_size=box_size)

    if title is not None:
        ax.set(title=title)

    plt.show()



def skill_vis(ax, env, policy, skills, rollouts_per_skill, skill_length, box_size=None):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts
    skill_trajectories = point_env_vis.collect_hier_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length)

    cmap = point_env_vis.get_cmap(len(skills))
    plot_all_skills(ax, cmap, skill_trajectories)

    #config_subplot(ax, box_size=box_size)


def plot_all_skills(ax, cmap, trajectories, alpha=0.5, linewidth=2):
    for skill_i in range(len(trajectories)):
        for i in range(len(trajectories[skill_i])):
            plot_trajectory(ax, trajectories[skill_i][i], cmap(skill_i), alpha=alpha, linewidth=linewidth)

    ax.plot3D(trajectories[0][0][0][0], trajectories[0][0][0][1], trajectories[0][0][0][2], marker='o', markersize=8, color='black', zorder=11)
    return ax


def plot_trajectory(ax, traj, color, alpha=0.5, linewidth=2):
    xs = [step[0] for step in traj]
    ys = [step[1] for step in traj]
    zs = [step[2] for step in traj]

    ax.plot3D(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth, zorder=10)


def rollout_single_trajectory():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    env = point_environment.PointEnv(step_size=0.1, box_size=1, dim=3)
    policy = RandomPyPolicy(env.time_step_spec(), env.action_spec())
    trajectory = []
    timestep = env.reset()
    for _ in range(150):
        trajectory.append(timestep.observation)
        actionstep = policy.action(timestep)
        timestep = env.step(actionstep.action)

    plot_trajectory(ax, trajectory, "green")
    ax.plot3D(0., 0., 0., marker='o', markersize=8,
              color='black', zorder=11)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)

    plt.show()





if __name__ == "__main__":
    rollout_single_trajectory()