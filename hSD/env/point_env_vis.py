import matplotlib.pyplot as plt

import tensorflow as tf
from env import point_environment
from env.maze import maze_env
from tf_agents.environments import tf_py_environment
from core.modules.rollout_drivers import collect_skill_trajectories


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


def config_subplot(ax, maze_type=None, extra_lim=0., title=None):
    if title is not None:
        ax.set_title(title, fontsize=14)

    if maze_type is not None:
        env = maze_env.MazeEnv(maze_type=maze_type)
        env.maze.plot(ax=ax)

        env_config = ENV_LIMS[maze_type]
        ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
        ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)
    else:  # default, point_env_limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #for p in ["left", "right", "top", "bottom"]:
    #    ax.spines[p].set_visible(False)


def discretize_continuous_space(min, max, num_points):
    step = (max-min) / num_points
    return [[min + step * x, min + step * y] for x in range(num_points) for y in range(num_points)]


def one_hots_for_num_skills(num_skills):
    return tf.one_hot(list(range(num_skills)), num_skills)


def skill_vis(ax, env, policy, skills, rollouts_per_skill, skill_length):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts
    skill_trajectories = collect_skill_trajectories(env, policy, skills, rollouts_per_skill, skill_length)

    cmap = get_cmap(len(skills))
    plot_all_skills(ax, cmap, skill_trajectories, alpha=0.2)

    config_subplot(ax,title="Skills")


def plot_all_skills(ax, cmap, trajectories, alpha=0.2, linewidth=2):
    for skill_i in range(len(trajectories)):
        for i in range(len(trajectories[skill_i])):
            traj = trajectories[skill_i][i]
            xs = [step[0] for step in traj]
            ys = [step[1] for step in traj]
            ax.plot(xs, ys, label="Skill #{}".format(skill_i), color=cmap(skill_i), alpha=alpha,
                    linewidth=linewidth, zorder=10)

    # mark starting point
    ax.plot(trajectories[0][0][0][0], trajectories[0][0][0][1], marker='o', markersize=8, color='black', zorder=11)

    return ax


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


def vis_skill_model(skill_model):
    pass


def vis_saved_policy(ax):
    env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv(step_size=0.05))
    saved_policy = tf.compat.v2.saved_model.load("logs")
    skill_vis(ax, saved_policy, 4, env, 10)


if __name__ == '__main__':
    pass