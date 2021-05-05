import matplotlib.pyplot as plt
import fixed_option_policy

import tensorflow as tf
import point_environment
from tf_agents.environments import tf_py_environment


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


def skill_vis(policy, num_skills, env, rollouts_per_skill):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts ->
    skill_trajectories = []
    for skill in range(num_skills):
        trajectories = rollout_skill(fixed_option_policy.FixedOptionPolicy(policy, skill, num_skills), env, rollouts_per_skill)
        skill_trajectories.append(trajectories)

    cmap = get_cmap(num_skills)
    plot_all_skills(cmap, skill_trajectories)
    plt.show()


def rollout_skill(skill_policy, env, num_rollouts):
    trajectories = []
    for _ in range(num_rollouts):
        traj = []
        time_step = env.reset()
        traj.append(time_step.observation.numpy().flatten())
        while not time_step.is_last():
            action_step = skill_policy.action(time_step)
            next_ts = env.step(action_step.action)
            traj.append(next_ts.observation.numpy().flatten())
            time_step = next_ts
        trajectories.append(traj)
    return trajectories


def plot_all_skills(cmap, trajectories, figsize=(5, 5), alpha=0.2, linewidth=2):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for skill in range(len(trajectories)):
        for i in range(len(trajectories[skill])):
            traj = trajectories[skill][i]
            xs = [step[0] for step in traj]
            ys = [step[1] for step in traj]
            ax.plot(xs, ys, label="Skill #{}".format(skill), color=cmap(skill), alpha=alpha,
                    linewidth=linewidth, zorder=10)

    # mark starting point
    ax.plot(trajectories[0][0][0][0], trajectories[0][0][0][1], marker='o', markersize=8, color='black', zorder=11)

    config_subplot(ax, title="Skills")

    return ax


def vis_saved_policy():
    env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    saved_policy = tf.compat.v2.saved_model.load("logs")
    skill_vis(saved_policy, 4, env, 10)


if __name__ == '__main__':
    vis_saved_policy()