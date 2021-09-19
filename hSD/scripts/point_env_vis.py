import matplotlib.pyplot as plt

import tensorflow as tf
from env.maze import maze_env
from core.modules import utils
from tf_agents.trajectories import time_step as ts



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


def config_subplot(ax, maze_type=None, box_size=None, extra_lim=0., title=None):
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



def discretize_continuous_space(min, max, num_points):
    step = (max-min) / num_points
    return [[min + step * x, min + step * y] for x in range(num_points) for y in range(num_points)]


def one_hots_for_num_skills(num_skills):
    return tf.one_hot(list(range(num_skills)), num_skills)


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


def plot_trajectory(ax, traj, color, alpha=0.5, linewidth=2):
    xs = [step[0] for step in traj]
    ys = [step[1] for step in traj]

    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=10)


def collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length):
    trajectories = [[] for _ in range(len(skills))]

    for i in range(len(skills)):
        for si in range(rollouts_per_skill):
            time_step = env.reset()
            traj, _ = rollout_skill_t_steps(env, policy, skills[i], time_step, skill_length)
            trajectories[i].append(traj)

    return trajectories


def rollout_skill_t_steps(env, policy, skill, time_step, t, state_norm=False):
    traj = []
    s_0 =  time_step.observation if state_norm else None
    for ti in range(t):
        traj.append(time_step.observation.numpy().flatten().tolist())
        aug_time_step = preprocess_time_step(time_step, skill, s_0)
        action_step = policy.action(aug_time_step)
        time_step = env.step(action_step.action)
    return traj, time_step


def preprocess_time_step(time_step, skill, s_norm):
    obs = time_step.observation - s_norm if s_norm is not None else time_step.observation
    return ts.TimeStep(time_step.step_type,
                       time_step.reward,
                       time_step.discount,
                       tf.concat([obs, tf.reshape(skill, (1, -1))], axis=-1))


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


if __name__ == '__main__':
    pass