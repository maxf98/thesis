import matplotlib.pyplot as plt
from core.policies import FixedOptionPolicy

import tensorflow as tf
import numpy as np
from env import point_environment
from tf_agents.environments import tf_py_environment
from core.skill_discriminators import UniformCategoricalDiscriminator
from core.skill_discriminators import OracleDiscriminator
from core.rollout_drivers import collect_skill_trajectories


def get_cmap(num_skills):
    if num_skills <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < num_skills <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis', num_skills)
    return cmap


def plot_trajectories(batch):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    cmap = plt.get_cmap('tab10')
    traj = []
    col = 0
    for ts in batch:
        if not ts.is_last():
            traj.append(ts.observation)
        else:
            xs = [ts[0][0] for ts in traj]
            ys = [ts[0][1] for ts in traj]
            ax.plot(xs, ys, color=cmap(col), linewidth=1)
            traj = []
            col += 1

    config_subplot(ax)
    plt.show()


def config_subplot(ax, title=None):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=14)


def discretize_continuous_space(min, max, num_points):
    step = (max-min) / num_points
    return [[min + step * x, min + step * y] for x in range(num_points) for y in range(num_points)]


def one_hots_for_num_skills(num_skills):
    return tf.one_hot(list(range(num_skills)), num_skills)


def skill_vis(ax, env, policy, skills, rollouts_per_skill, skill_length):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts ->
    skill_trajectories = collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length)

    cmap = get_cmap(len(skills))
    plot_all_skills(ax, cmap, skill_trajectories, alpha=0.2)


def plot_all_skills(ax, cmap, trajectories, alpha=0.2, linewidth=2):
    for skill_i in range(len(trajectories)):
        for i in range(len(trajectories[skill_i])):
            traj = trajectories[skill_i][i]
            xs = [step[0] for step in traj]
            ys = [step[1] for step in traj]
            ax.plot(xs, ys, label="Skill #{}".format(skill_i), color=cmap(skill_i), alpha=alpha,
                    linewidth=linewidth, zorder=10)

    # mark starting point
    ax.plot([0], [0], marker='o', markersize=8, color='black', zorder=11)

    config_subplot(ax, title="Skills")

    return ax


def categorical_discrim_heatmap(ax, discriminator):
    P = 50
    points = tf.constant([[i/P, j/P] for i in range(-P, P, 1) for j in range(-P, P, 1)])
    pred = discriminator.call(points, return_probs=True)
    s, p = tf.reshape(tf.argmax(pred, axis=-1), (2*P, 2*P)), tf.reshape(tf.reduce_max(pred, axis=-1), (2*P, 2*P))
    s, p = s.numpy(), p.numpy()

    cmap = get_cmap(discriminator.latent_dim)
    ax.imshow(s, cmap, interpolation='none', alpha=p)

    ax.plot([50], [50], marker='o', markersize=8, color='black', zorder=11)

    return ax


def vis_saved_policy(ax):
    env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    saved_policy = tf.compat.v2.saved_model.load("logs")
    skill_vis(ax, saved_policy, 4, env, 10)


if __name__ == '__main__':
    pass