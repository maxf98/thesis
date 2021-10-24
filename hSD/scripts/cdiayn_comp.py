import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt

from env import point_environment
from scripts import point_env_vis

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_py_policy import RandomPyPolicy

from core.modules import utils


def compare_cont_discrete_diayn():
    # visualise both, compare convergence behaviour in terms of intrinsic rewards and discriminator accuracy on same axis
    # after how many epochs?
    # this function is pretty much just for that one thesis plot...

    disc_policy_dir = "../logs/diayn/thesis/diayn/policies/policy_0"
    cont_policy_dir = "../logs/diayn/thesis/cdiayn/policies/policy_0"

    disc_policy = tf.compat.v2.saved_model.load(disc_policy_dir)
    cont_policy = tf.compat.v2.saved_model.load(cont_policy_dir)

    env = TFPyEnvironment(point_environment.PointEnv(step_size=0.1, box_size=1))

    num_skills = 8

    cont_skills = utils.discretize_continuous_space(-1, 1, 3, 2)
    disc_skills = tf.one_hot(list(range(num_skills)), num_skills)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

    point_env_vis.skill_vis(ax1, env, disc_policy, disc_skills, 3, skill_length=25)
    ax1.set(title="DIAYN")

    point_env_vis.skill_vis(ax2, env, cont_policy, cont_skills, 3, skill_length=25)
    ax2.set(title="cDIAYN")

    plt.show()


def vis_saved_policy(ax, policy_dir, cont=True, title=None, env=None, skill_length=25, step_size=0.1, box_size=None):
    policy = tf.compat.v2.saved_model.load(policy_dir)
    if env is None:
        env = TFPyEnvironment(point_environment.PointEnv(step_size=step_size, box_size=box_size))
    if cont:
        skills = utils.discretize_continuous_space(-1, 1, 3, 2)
    else:
        NUM_SKILLS = 8
        skills = tf.one_hot(list(range(NUM_SKILLS)), NUM_SKILLS)

    point_env_vis.skill_vis(ax, env, policy, skills, 3, skill_length=skill_length, box_size=box_size)

    if title is not None:
        ax.set(title=title)



def vis_entropy_policies():
    e01 = "../logs/diayn/thesis/entropy/cdiayn-01/policies/policy_0"
    e05 = "../logs/diayn/thesis/entropy/cdiayn-05/policies/policy_0"
    e1 = "../logs/diayn/thesis/entropy/cdiayn-1/policies/policy_0"
    e2 = "../logs/diayn/thesis/entropy/cdiayn-2/policies/policy_0"
    e5 = "../logs/diayn/thesis/entropy/cdiayn-5/policies/policy_0"
    e10 = "../logs/diayn/thesis/cdiayn/policies/policy_0"

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    vis_saved_policy(ax1, e01, cont=True, title="alpha=10")
    vis_saved_policy(ax2, e05, cont=True, title="alpha=2")
    vis_saved_policy(ax3, e1, cont=True, title="alpha=1")
    vis_saved_policy(ax4, e2, cont=True, title="alpha=0.5")
    vis_saved_policy(ax5, e5, cont=True, title="alpha=0.2")
    vis_saved_policy(ax6, e10, cont=True, title="alpha=0.1")

    plt.tight_layout()
    plt.show()


def vis_local_optima():
    epoch10 = "../logs/diayn/thesis/cdiayn/0/policy_10"
    epoch20 = "../logs/diayn/thesis/cdiayn/0/policy_30"
    epoch30 = "../logs/diayn/thesis/cdiayn/0/policy_60"
    epoch40 = "../logs/diayn/thesis/cdiayn/0/policy_70"

    hepoch10 = "../logs/diayn/thesis//entropy/cdiayn-1/0/policy_10"
    hepoch20 = "../logs/diayn/thesis//entropy/cdiayn-1/0/policy_20"
    hepoch30 = "../logs/diayn/thesis//entropy/cdiayn-1/0/policy_30"
    hepoch40 = "../logs/diayn/thesis//entropy/cdiayn-1/0/policy_40"

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    vis_saved_policy(ax1, epoch10, cont=True, title="epoch 10 (alpha = 0.1)")
    vis_saved_policy(ax2, epoch20, cont=True, title="epoch 30")
    vis_saved_policy(ax3, epoch30, cont=True, title="epoch 60")
    vis_saved_policy(ax4, epoch40, cont=True, title="epoch 70")
    vis_saved_policy(ax5, hepoch10, cont=True, title="epoch 10 (alpha=1)")
    vis_saved_policy(ax6, hepoch20, cont=True, title="epoch 20")
    vis_saved_policy(ax7, hepoch30, cont=True, title="epoch 30")
    vis_saved_policy(ax8, hepoch40, cont=True, title="epoch 40")

    plt.tight_layout()
    plt.show()


def vis_run():
    dir = "../logs/diayn/thesis/rewardscaling"
    policy_dir = os.path.join(dir, "0/policies/policy_100")
    discrim_acc = np.load(os.path.join(dir, "0/vis/discrim_acc.npy"))
    ir = np.load(os.path.join(dir, "0/vis/intrinsic_rewards.npy"))

    alpha = [1000/(i + 100) for i in range(len(ir))]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    vis_saved_policy(ax1, policy_dir, cont=True, title="epoch 100 (alpha-annealing)")
    ax1.set_aspect('equal', adjustable='box')

    ax2.plot(range(len(discrim_acc)), discrim_acc, color='lightblue', linewidth=3)
    ax2.set(title='Skill model accuracy')

    ax3.plot(range(len(ir)), ir, color='green', linewidth=3)
    ax3.set(title='intrinsic reward')

    ax4.plot(range(len(ir)), alpha, color='red', linewidth=3)
    ax4.set(title='alpha')

    #plt.tight_layout()
    plt.show()


def vis_out_of_distribution_skills():
    # just increase size of point environment and shift start-state to (-1, 0)
    policy_dir = "../logs/diayn/thesis/entropy/cdiayn-1/policies/policy_0"
    box_size = 2
    env = TFPyEnvironment(point_environment.PointEnv(step_size=0.1, box_size=box_size))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
    vis_saved_policy(ax1, policy_dir, cont=True, title=None, env=env, skill_length=20, box_size=box_size)

    env = point_environment.PointEnv(step_size=0.1, box_size=box_size)
    env.set_start_state((-1.75, 1.75))
    env = TFPyEnvironment(env)

    vis_saved_policy(ax2, policy_dir, cont=True, title=None, env=env, skill_length=30, box_size=box_size)

    plt.show()


def sequence_state():
    policy_dir = "../logs/diayn/thesis/entropy/cdiayn-1/policies/policy_0"

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    vis_saved_policy(ax1, policy_dir, box_size=1)
    rollout_skill_sequence(ax2, policy_dir, box_size=1, s_norm=False)
    rollout_skill_sequence(ax3, policy_dir, box_size=3, s_norm=True)

    plt.show()


def rollout_skill_sequence(ax, policy_dir, box_size, s_norm):
    policy = tf.compat.v2.saved_model.load(policy_dir)
    env = TFPyEnvironment(point_environment.PointEnv(step_size=0.1, box_size=box_size))
    skills = utils.discretize_continuous_space(-1, 1, 3, 2)
    cmap = point_env_vis.get_cmap(len(skills))

    state_seq = []
    time_step = env.reset()
    for skill in skills[:5]:
        skill_seq, time_step = point_env_vis.rollout_skill_t_steps(env, policy, skill, time_step, 20, state_norm=s_norm)
        ax.plot(skill_seq[0][0], skill_seq[0][1], marker='o', markersize=3, color='black', zorder=11)
        state_seq.append([skill_seq])

    point_env_vis.plot_all_skills(ax, cmap, state_seq, alpha=1)
    point_env_vis.config_subplot(ax, box_size=box_size)


def vis_rand_pol_states(ax, step_size=0.1, rollout_length=10, num_rollouts=100, keep_every=1, color='blue', alpha=0.8):
    box_size = step_size * rollout_length
    env = point_environment.PointEnv(step_size=step_size, box_size=box_size)
    policy = RandomPyPolicy(time_step_spec=env.time_step_spec(), action_spec=env.action_spec())

    states = []

    for _ in range(num_rollouts):
        timestep = env.reset()

        for _ in range(rollout_length):
            action_step = policy.action(timestep)
            timestep = env.step(action_step.action)
            states.append(timestep.observation)

    states = states[::keep_every]
    xs, ys = [s[0] for s in states], [s[1] for s in states]
    ax.scatter(xs, ys, color=color, alpha=alpha, s=0.5)

    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    ax.set_xticks([-box_size, 0., box_size])
    ax.set_yticks([-box_size, 0., box_size])
    ax.set_aspect('equal', adjustable='box')


def skill_pol_coverage(ax, policy, cont=True, step_size=0.1, skill_length=10, num_rollouts=100, keep_every=1, color='blue', alpha=0.8):
    box_size = 1.
    env = TFPyEnvironment(point_environment.PointEnv(step_size=step_size, box_size=box_size))

    if cont:
        skill_dim = 2
        distr = tfd.Uniform(low=[-1.] * skill_dim, high=[1.] * skill_dim)
    else:
        skill_dim = 8
        distr = tfd.OneHotCategorical(logits=tf.ones(skill_dim), dtype=tf.float32)

    skills = distr.sample(num_rollouts)
    trajs = point_env_vis.collect_skill_trajectories(env, policy, skills, 1, skill_length)
    points = np.reshape(trajs, (num_rollouts * skill_length, 2))
    points = points[::keep_every]
    xs, ys = [s[0] for s in points], [s[1] for s in points]
    ax.scatter(xs, ys, color=color, alpha=alpha, s=0.5)

    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    ax.set_xticks([-box_size, 0., box_size])
    ax.set_yticks([-box_size, 0., box_size])
    ax.set_aspect('equal', adjustable='box')


def comp_rand_pol_coverage():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    vis_rand_pol_states(ax1, step_size=0.01, rollout_length=100, num_rollouts=100)
    ax1.set_title("δ = 0.01, T = 100")
    vis_rand_pol_states(ax2, step_size=0.1, rollout_length=10, num_rollouts=100)
    ax2.set_title("δ = 0.1, T = 10")
    vis_rand_pol_states(ax3, step_size=1., rollout_length=1, num_rollouts=100)
    ax3.set_title("δ = 1, T = 1")

    plt.show()


def comp_traj_length():
    policy_10_dir = "../logs/traj_length/l1/0/policies/policy_10"
    policy_10 = tf.compat.v2.saved_model.load(policy_10_dir)
    policy_20_dir = "../logs/traj_length/l2-rb/0/policies/policy_10"
    policy_20 = tf.compat.v2.saved_model.load(policy_20_dir)
    policy_100_dir = "../logs/traj_length/l3-rb/0/policies/policy_10"
    policy_100 = tf.compat.v2.saved_model.load(policy_100_dir)

    policy_20_dir = "../logs/traj_length/l2-rb/0/policies/policy_30"
    policy_20_f = tf.compat.v2.saved_model.load(policy_20_dir)
    policy_100_dir = "../logs/traj_length/l3-rb/0/policies/policy_100"
    policy_100_f = tf.compat.v2.saved_model.load(policy_100_dir)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

    vis_saved_policy(ax1, policy_10_dir, skill_length=10, step_size=0.1, box_size=1.)
    vis_saved_policy(ax2, policy_20_dir, skill_length=25, step_size=0.04, box_size=1.)
    vis_saved_policy(ax3, policy_100_dir, skill_length=100, step_size=0.01, box_size=1.)

    skill_pol_coverage(ax4, policy_10, step_size=0.1, skill_length=10, num_rollouts=100, keep_every=1, color='green', alpha=0.5)
    skill_pol_coverage(ax5, policy_20, step_size=0.04, skill_length=25, num_rollouts=100, keep_every=2, color='green', alpha=0.5)
    skill_pol_coverage(ax6, policy_100, step_size=0.01, skill_length=100, num_rollouts=100, keep_every=10, color='green', alpha=0.5)

    vis_rand_pol_states(ax4, step_size=0.1, rollout_length=10, num_rollouts=100, keep_every=1, color='blue', alpha=0.5)
    vis_rand_pol_states(ax5, step_size=0.04, rollout_length=25, num_rollouts=100, keep_every=2, color='blue', alpha=0.5)
    vis_rand_pol_states(ax6, step_size=0.01, rollout_length=100, num_rollouts=100, keep_every=10, color='blue', alpha=0.5)

    skill_pol_coverage(ax7, policy_10, step_size=0.1, skill_length=10, num_rollouts=100, keep_every=1, color='green', alpha=0.5)
    skill_pol_coverage(ax8, policy_20_f, step_size=0.04, skill_length=25, num_rollouts=100, keep_every=2, color='green', alpha=0.5)
    skill_pol_coverage(ax9, policy_100_f, step_size=0.01, skill_length=100, num_rollouts=100, keep_every=10, color='green', alpha=0.5)

    vis_rand_pol_states(ax7, step_size=0.1, rollout_length=10, num_rollouts=100, keep_every=1, color='blue', alpha=0.5)
    vis_rand_pol_states(ax8, step_size=0.04, rollout_length=25, num_rollouts=100, keep_every=2, color='blue', alpha=0.5)
    vis_rand_pol_states(ax9, step_size=0.01, rollout_length=100, num_rollouts=100, keep_every=10, color='blue', alpha=0.5)

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()


if __name__ == '__main__':
    #compare_cont_discrete_diayn()
    """
    policy_dir = "../logs/diayn/thesis/hiercomp/flat/0/policies/policy_100"

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vis_saved_policy(ax, policy_dir, cont=True, title="Hello", skill_length=100, box_size=5)
    plt.show()
    """

    comp_traj_length()

    #vis_entropy_policies()

    #vis_local_optima()

    #vis_run()

    #vis_out_of_distribution_skills()

    #sequence_state()
    #vis_out_of_distribution_skills()
    #vis_out_of_distribution_skills()
