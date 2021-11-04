import matplotlib.pyplot as plt

import tensorflow as tf
from env.maze import maze_env

from core.modules import rollout_drivers, utils



ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5))
)


def get_cmap(num_skills):
    if num_skills <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < num_skills <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis', num_skills)
    return cmap


def config_subplot(ax, maze_type="square_bottleneck", box_size=None, extra_lim=0., title=None):
    if title is not None:
        ax.set_title(title, fontsize=14)

    if maze_type is not None:
        env = maze_env.MazeEnv(maze_type=maze_type)
        env.maze.plot(ax=ax)

        env_config = ENV_LIMS[maze_type]
        ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
        ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)
    else:  # default, point_env_limits
        if box_size is None:
            box_size = 1
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)
        ax.set_xticks([-box_size, 0., box_size])
        ax.set_yticks([-box_size, 0., box_size])
        ax.set_aspect('equal', adjustable='box')

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #for p in ["left", "right", "top", "bottom"]:
    #    ax.spines[p].set_visible(False)


def skill_vis(ax, env, policy, skills, rollouts_per_skill, skill_length, box_size=None):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts
    skill_trajectories = collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length)

    cmap = get_cmap(len(skills))
    plot_all_skills(ax, cmap, skill_trajectories)

    config_subplot(ax, box_size=box_size)


def plot_all_skills(ax, cmap, trajectories, alpha=0.5, linewidth=2):
    for skill_i in range(len(trajectories)):
        for i in range(len(trajectories[skill_i])):
            plot_trajectory(ax, trajectories[skill_i][i], cmap(skill_i), alpha=alpha, linewidth=linewidth)

    ax.plot(trajectories[0][0][0][0], trajectories[0][0][0][1], marker='o', markersize=8, color='black', zorder=11)
    return ax


def plot_trajectory(ax, traj, color, alpha=0.5, linewidth=2, label=None):
    xs = [step[0] for step in traj]
    ys = [step[1] for step in traj]

    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=10, label=label)


def collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length):
    trajectories = [[] for _ in range(len(skills))]

    for i in range(len(skills)):
        for si in range(rollouts_per_skill):
            time_step = env.reset()
            cur_traj = []
            rollout_drivers.rollout_skill_trajectory(time_step, env, policy, rollout_drivers.preprocess_time_step, [cur_traj.append],skills[i], skill_length, return_aug_obs=False, s_norm=True)
            trajectories[i].append(extract_obs(cur_traj))

    return trajectories


def extract_obs(traj):
    return [tf.reshape(utils.hide_goal(subtraj.observation), (-1)) for subtraj in traj]


def categorical_discrim_heatmap(ax, discriminator):
    P = 50
    points = tf.constant([[i/P, j/P] for i in range(-P, P, 1) for j in range(-P, P, 1)])
    pred = discriminator.call(points)
    s, p = tf.reshape(tf.argmax(pred, axis=-1), (2*P, 2*P)), tf.reshape(tf.reduce_max(pred, axis=-1), (2*P, 2*P))
    s, p = s.numpy(), p.numpy()

    cmap = get_cmap(discriminator.output_dim)
    ax.imshow(s, cmap, interpolation='none', alpha=p)

    ax.plot([50], [50], marker='o', markersize=8, color='black', zorder=11)

    return ax


def cont_diayn_skill_heatmap(ax, discriminator):
    # technically need one axis for each dimension of the skill model... for now we only visualise the first dimension
    P = 50
    points = tf.constant([[i / P, j / P] for i in range(-P, P, 1) for j in range(-P, P, 1)])
    pred_distr = discriminator.call(points)  # will automatically sample from out_distr for each point
    s = tf.reshape(pred_distr.mean[0], (2*P, 2*P))

    ax.imshow(s)

    return ax


def per_skill_collect_rollouts(policy, vis_skill_set, skill_length, env):
    fig, axes = plt.subplots(nrows=len(vis_skill_set) // 4 + 1, ncols=4, figsize=(8, 8))
    for i, skill in enumerate(vis_skill_set):
        skill_vis(axes[i], env, policy, [skill], 10, skill_length)
        axes[i].set(title=skill)
        axes[i].set_aspect('equal', adjustable='box')

    return fig


if __name__ == '__main__':
    pass