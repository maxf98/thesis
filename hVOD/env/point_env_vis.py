import matplotlib.pyplot as plt
from utils import fixed_option_policy

import tensorflow as tf
from env import point_environment
from tf_agents.environments import tf_py_environment
from core.diayn.diayn import DIAYNDiscriminator
from core.diayn.diayn import OracleDiscriminator


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


def skill_vis(ax, policy, num_skills, env, rollouts_per_skill):
    # collect rollouts w/ policy (store unaugmented observations...) -> visualise rollouts ->
    skill_trajectories = []
    for skill in range(num_skills):
        trajectories = rollout_skill(fixed_option_policy.FixedOptionPolicy(policy, skill, num_skills), env, rollouts_per_skill)
        skill_trajectories.append(trajectories)

    cmap = get_cmap(num_skills)
    plot_all_skills(ax, cmap, skill_trajectories)

# refactor skill_vis so it just takes a policy and a set of skills from which to generate skill policies
def cont_skill_vis(ax, policy, env, num_steps):
    # assumes skill space of [[-1, -1],[1, 1]]
    dmin, dmax = -1, 1
    points_per_axis = 3

    skill_trajectories = []
    for x in range(points_per_axis):
        for y in range(points_per_axis):
            skill = [2 * x / points_per_axis - 1, 2 * y / points_per_axis - 1]
            skill_policy = fixed_option_policy.FixedOptionPolicyCont(policy, skill)
            trajectories = rollout_skill(skill_policy, env, 1, num_steps=num_steps)
            skill_trajectories.append(trajectories)

    cmap = get_cmap(points_per_axis * points_per_axis)
    plot_all_skills(ax, cmap, skill_trajectories, alpha=0.8)


def rollout_skill(skill_policy, env, num_rollouts, num_steps = 50):
    trajectories = []
    for _ in range(num_rollouts):
        traj = []
        time_step = env.reset()
        traj.append(time_step.observation.numpy().flatten())
        for _ in range(num_steps):
            action_step = skill_policy.action(time_step)
            next_ts = env.step(action_step.action)
            traj.append(next_ts.observation.numpy().flatten())
            time_step = next_ts
        trajectories.append(traj)
    return trajectories


def plot_all_skills(ax, cmap, trajectories, alpha=0.2, linewidth=2):
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


def heatmap(ax, discriminator):
    P = 50
    points = tf.constant([[i/P, j/P] for i in range(-P, P, 1) for j in range(-P, P, 1)])
    pred = discriminator.call(points)

    vals = tf.reshape(tf.argmax(pred, axis=-1), (2*P, 2*P))
    cmap = get_cmap(discriminator.num_skills)
    ax.scatter(points[:,0], points[:,1], c=vals, cmap=cmap, s=10, marker='o')

    ax.plot(0, 0, marker='o', markersize=8, color='black', zorder=11)

    config_subplot(ax, "Discriminator heatmap")

    return ax


def latent_skill_vis(ax, decoder: tf.keras.models.Model):
    # assumes skill space of [[-1, -1],[1, 1]]
    points_per_axis = 3

    skill_trajectories = []
    for x in range(points_per_axis):
        for y in range(points_per_axis):
            skill = [2 * x / points_per_axis - 1, 2 * y / points_per_axis - 1]
            delta_s = decoder.call(tf.constant([skill]))
            skill_trajectories.append([[0., 0.], delta_s])

    cmap = get_cmap(points_per_axis * points_per_axis)
    plot_all_skills(ax, cmap, skill_trajectories, alpha=0.8)


def vis_saved_policy(ax):
    env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    saved_policy = tf.compat.v2.saved_model.load("logs")
    skill_vis(ax, saved_policy, 4, env, 10)


def vis_saved_discrim(ax):
    discriminator = DIAYNDiscriminator(4, load_from="logs/discriminator")
    heatmap(ax, discriminator)


def vis_oracle_discrim():
    fig, ax = plt.subplots()
    heatmap(ax, OracleDiscriminator((), (), 4))
    plt.show()


if __name__ == '__main__':
    vis_oracle_discrim()