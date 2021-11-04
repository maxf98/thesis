import os

import gin
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt

from env import point_environment
from scripts import point_env_vis
import launcher

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_py_policy import RandomPyPolicy

from core.modules import utils
from core.modules import rollout_drivers

from env import skill_environment
from env.maze import maze_env, mazes

from mpl_toolkits import axes_grid1


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


def vis_saved_policy(ax, policy_dir, cont=True, title=None, env=None, skill_length=25, step_size=0.1, box_size=None, skill_dim=2, skill_samples=3):
    policy = tf.compat.v2.saved_model.load(policy_dir)
    if env is None:
        env = TFPyEnvironment(point_environment.PointEnv(step_size=step_size, box_size=box_size))
    if cont:
        skills = utils.discretize_continuous_space(-1, 1, skill_samples, skill_dim)
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

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    fig, ((ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(2, 3)
    """
    vis_saved_policy(ax1, policy_10_dir, skill_length=10, step_size=0.1, box_size=1.)
    vis_saved_policy(ax2, policy_20_dir, skill_length=25, step_size=0.04, box_size=1.)
    vis_saved_policy(ax3, policy_100_dir, skill_length=100, step_size=0.01, box_size=1.)
    """
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

    fontdict = {'fontsize': 8}
    ax4.set_title("δ = 0.1, T = 10, E=10", fontdict=fontdict)
    ax5.set_title("δ = 0.04, T = 25, E=10", fontdict=fontdict)
    ax6.set_title("δ = 0.01, T = 100, E=10", fontdict=fontdict)

    ax7.set_title("δ = 0.1, T = 10, E=10", fontdict=fontdict)
    ax8.set_title("δ = 0.04, T = 25, E=30", fontdict=fontdict)
    ax9.set_title("δ = 0.01, T = 100, E=100", fontdict=fontdict)

    for ax in fig.get_axes():
        ax.label_outer()

    fig.savefig("../screenshots/exploration_vs_trajlength")


def vis_reach_goal_state_behaviour():
    policy_dir1 = "../logs/traj_length/l1e/0/policies/policy_10"
    policy_dir2 = "../logs/traj_length/l1e/0/policies/policy_30"
    policy_dir3 = "../logs/traj_length/l1e/0/policies/policy_50"
    discrim_acc = np.load("../logs/traj_length/l1e/0/stats/discrim_acc.npy")
    ir = np.load("../logs/traj_length/l1e/0/stats/intrinsic_rewards.npy")

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:])
    ax4 = fig.add_subplot(gs[1, :3])
    ax5 = fig.add_subplot(gs[1, 3:])

    vis_saved_policy(ax1, policy_dir1, cont=True, title="10 epochs", skill_length=10, box_size=1)
    vis_saved_policy(ax2, policy_dir2, cont=True, title="30 epochs", skill_length=10, box_size=1)
    vis_saved_policy(ax3, policy_dir3, cont=True, title="50 epochs", skill_length=10, box_size=1)

    ax4.plot(range(len(discrim_acc)), discrim_acc, color='lightblue', linewidth=3)
    ax4.set(title='Discriminator Accuracy')

    ax5.plot(range(len(ir)), ir, color='green', linewidth=3)
    ax5.set(title='Intrinsic Reward')

    #fig.savefig("../screenshots/reachgoalstatebehaviour")
    plt.show()


def vis_hierarchy_policy_init_from_other_agent():
    base_config_path = "/home/max/RL/thesis/hSD/logs/traj_length/l1e/config3.gin"
    config_path = "/home/max/RL/thesis/hSD/logs/traj_length/hier_l1e/config2.gin"
    gin.parse_config_file(base_config_path)
    base_envs, base_agents = launcher.hierarchical_skill_discovery(config_path=base_config_path)
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)

    base_env, l1_agent = base_envs[0], base_agents[0]
    l2_agent = agents[0]
    l1_env = skill_environment.SkillEnv(base_env, l1_agent.policy_learner.policy, l1_agent.rollout_driver.skill_length, l1_agent.skill_dim)

    skills = utils.discretize_continuous_space(-1, 1, 3, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    #point_env_vis.skill_vis(ax1, TFPyEnvironment(base_env), l1_agent.policy_learner.policy, skills, 3, skill_length=l1_agent.rollout_driver.skill_length, box_size=1)
    point_env_vis.skill_vis(ax2, TFPyEnvironment(l1_env), l2_agent.policy_learner.policy, skills, 3, skill_length=l2_agent.rollout_driver.skill_length, box_size=1)

    plt.show()


def vis_hierarchy_policy():
    hier_dir = "/home/max/RL/thesis/hSD/logs/traj_length/hier3/"
    config_path = "/home/max/RL/thesis/hSD/logs/traj_length/hier3/config.gin"
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)

    env, l1_env = TFPyEnvironment(envs[0]), TFPyEnvironment(envs[1])
    l1_policy, l2_policy = agents[0].policy_learner.policy, agents[1].policy_learner.policy
    l2_policy = tf.compat.v2.saved_model.load("/home/max/RL/thesis/hSD/logs/traj_length/hier3/1/policies/policy_9")

    skills = utils.discretize_continuous_space(-1, 1, 3, 2)

    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1.5, 1])
    ax1, ax2, ax3 = subfigs[0].subplots(1, 3)
    ax4, ax5 = subfigs[1].subplots(1, 2)

    point_env_vis.skill_vis(ax1, env, l1_policy, skills, 3, skill_length=10, box_size=1)
    ax1.set_title(r"$T=10$")
    point_env_vis.skill_vis(ax2, l1_env, l2_policy, skills, 3, skill_length=10, box_size=4)
    ax2.set_title(r"$T=(10,10)$")
    flat_dir = "../logs/traj_length/hier3flat/"
    policy_dir = "../logs/traj_length/hier3flat/0/policies/policy_15"

    vis_saved_policy(ax3, policy_dir, skill_length=100, step_size=0.1, box_size=4, skill_dim=2, skill_samples=3)
    ax3.set_title(r"$T=100$")

    dah1 = np.load(os.path.join(hier_dir, "0/stats/discrim_acc.npy"))
    dah = np.load(os.path.join(hier_dir, "1/stats/discrim_acc.npy"))
    daf = np.load(os.path.join(flat_dir, "0/stats/discrim_acc.npy"))
    irh1 = np.load(os.path.join(hier_dir, "0/stats/intrinsic_rewards.npy"))
    irh = np.load(os.path.join(hier_dir, "1/stats/intrinsic_rewards.npy"))
    irf = np.load(os.path.join(flat_dir, "0/stats/intrinsic_rewards.npy"))

    #ax4.plot(range(15), dah1[:15], color="red", linewidth=2, alpha=0.6)
    ax4.plot(range(15), dah[:15], color="blue", linewidth=2, alpha=0.6)
    ax4.plot(range(15), daf[:15], color="green", linewidth=2, alpha=0.6)

    ax4.set_title("Discriminator Accuracy")
    ax4.set_ylabel(r"$E[q_\phi(z|s)]$")
    ax4.set_xlabel("epoch")

    #ax5.plot(range(15), irh1[:15], color="red", label=r"$T=10$", linewidth=2, alpha=0.6)
    ax5.plot(range(15), irh[:15], color="blue", label=r"$T=(10,10)$", linewidth=2, alpha=0.6)
    ax5.plot(range(15), irf[:15], color="green", label=r"$T=100$", linewidth=2, alpha=0.6)
    ax5.set_title("Intrinsic Reward")
    ax5.set_ylabel(r"$E[r_z(s)]$")
    ax5.set_xlabel("epoch")
    ax5.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))

    fig.tight_layout()

    fig.savefig("../screenshots/2dNavHiercomp")

    plt.show()


def vis_hierarchy_run(agent_path):
    envs, agents = load_agent(agent_path)

    fig, axes = plt.subplots(1, len(agents))

    skills = utils.discretize_continuous_space(-1, 1, 2, 2)

    for i in range(len(agents)):
        ax, env, policy = axes[i], envs[i], agents[i].policy_learner.policy
        skill_length = agents[i].rollout_driver.skill_length
        point_env_vis.skill_vis(ax, env, policy, skills=skills, rollouts_per_skill=1, skill_length=skill_length)

    plt.show()




def load_agent(path):
    config_path = find_config_file(path)
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)
    return envs, agents


def maze_exploration():
    maze_dir = "../logs/maze"
    envs, agents = load_agent(maze_dir)

    l1_env = envs[1]
    rand_pol = RandomPyPolicy(l1_env.time_step_spec(), l1_env.action_spec())
    policy = agents[0].policy_learner.policy

    skills = utils.discretize_continuous_space(-1, 1, 2, 2)

    fig, ax = plt.subplots()
    point_env_vis.skill_vis(ax, l1_env, rand_pol, skills, 1, 15)

    plt.show()


def skill_dim_impact():
    two_one_dir = "../iglogs/2dnav-1ds/"
    two_two_dir = "../iglogs/2dnav-2ds/"
    two_three_dir = "../iglogs/pointoverspec/"
    three_two_dir = "../iglogs/3dnav-2ds/0/policies/policy_10"

    two_one_pol_dir = os.path.join(two_one_dir, "0/policies/policy_10")
    two_two_pol_dir = os.path.join(two_two_dir, "0/policies/policy_20")
    two_three_pol_dir = os.path.join(two_three_dir, "0/policies/policy_30")

    da1 = np.load(os.path.join(two_one_dir, "0/stats/discrim_acc.npy"))
    ir1 = np.load(os.path.join(two_one_dir, "0/stats/intrinsic_rewards.npy"))
    da2 = np.load(os.path.join(two_two_dir, "0/stats/discrim_acc.npy"))
    ir2 = np.load(os.path.join(two_two_dir, "0/stats/intrinsic_rewards.npy"))
    da3 = np.load(os.path.join(two_three_dir, "0/stats/discrim_acc.npy"))
    ir3 = np.load(os.path.join(two_three_dir, "0/stats/intrinsic_rewards.npy"))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:])
    ax4 = fig.add_subplot(gs[1, :3])
    ax5 = fig.add_subplot(gs[1, 3:])

    vis_saved_policy(ax1, two_one_pol_dir, cont=True, title=r"$|\mathcal{Z}|=1$", skill_length=20, box_size=2, skill_dim=1, skill_samples=4)
    vis_saved_policy(ax2, two_two_pol_dir, cont=True, title=r"$|\mathcal{Z}|=2$", skill_length=20, box_size=2, skill_dim=2, skill_samples=3)
    vis_saved_policy(ax3, two_three_pol_dir, cont=True, title=r"$|\mathcal{Z}|=3$", skill_length=20, box_size=2, skill_dim=3, skill_samples=2)

    #ax4.plot(range(30), da1[:30], color="blue", linewidth=2)
    ax4.plot(range(30), da2[:30], color="green", linewidth=2)
    ax4.plot(range(30), da3[:30], color="red", linewidth=2)
    ax4.set_ylabel(r"$E[q_\phi(z|s)]$")
    ax4.set_xlabel("epoch")

    ax5.plot(range(30), ir1[:30], color="blue", label=r"$|\mathcal{Z}|=1$", linewidth=2)
    ax5.plot(range(30), ir2[:30], color="green", label=r"$|\mathcal{Z}|=2$", linewidth=2)
    ax5.plot(range(30), ir3[:30], color="red", label=r"$|\mathcal{Z}|=3$", linewidth=2)
    ax5.set_ylabel(r"$E[r_z(s)]$")
    ax5.set_xlabel("epoch")
    ax5.legend(loc="right")

    plt.show()


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def vis_skill_smoothness():
    # skill interpolation and discriminator smoothness...
    agent_dir = "../logs/traj_length/hier3/"
    config_path = os.path.join(agent_dir, "config.gin")
    gin.parse_config_file(config_path)
    envs, agents = launcher.hierarchical_skill_discovery(config_path=config_path)

    base_env = TFPyEnvironment(envs[0])
    policy = agents[0].policy_learner.policy
    #policy = tf.compat.v2.saved_model.load("/Users/maxfest/RL/thesis/hSD/logs/traj_length/l3-rb/0/policies/policy_15")

    #skills = [[-1., 1.], [-0.6, 1.], [-0.2, 1.], [0.2, 1.], [0.6, 1.], [1., 1.], [1., 0.6], [1., 0.2], [1.0, -0.2], [1.0, -0.6], [1., -1.]]
    skills = [[-0.3, 1.], [0.3, 1.], [1., 1.], [1., 0.3], [1.0, -0.3], [1., -1.], [-0.2, -0.2], [-0.4, -0.4], [-0.6, -0.6]]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax2.remove()

    cmap = point_env_vis.get_cmap(len(skills))

    for i in range(len(skills)):
        time_step = base_env.reset()
        traj, _ = point_env_vis.rollout_skill_t_steps(base_env, policy, skills[i], time_step=time_step, t=15, state_norm=True)
        point_env_vis.plot_trajectory(ax1, traj, cmap(i), alpha=1, label=skills[i])

    ax1.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))

    ax1.plot(0.0, 0.0, marker='o', markersize=4, color='black', zorder=11)

    point_env_vis.config_subplot(ax1, box_size=1.0)

    vis_discriminator(agents[0].skill_model)

    plt.show()


def vis_discriminator(discriminator):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    P = 100
    points = tf.constant([[i / P, j / P] for i in range(-P, P, 1) for j in range(-P, P, 1)])
    pred_distr = discriminator.call(points)  # will automatically sample from out_distr for each point
    pred = pred_distr.sample()
    pred_x, pred_y = tf.split(pred, [1, 1], axis=-1)
    s1 = tf.reshape(pred_x, (2 * P, 2 * P))
    s2 = tf.reshape(pred_y, (2 * P, 2 * P))

    ax1.imshow(s1, extent=[-1., 1., -1., 1.])
    plot = ax2.imshow(s2, extent=[-1., 1., -1., 1.])

    ax1.set_title(r"$z_0$")
    ax1.set_xticks([-1, 0., 1])
    ax1.set_yticks([-1, 0., 1])

    ax2.set_title(r"$z_1$")
    ax2.set_xticks([-1, 0., 1])
    ax2.set_yticks([-1, 0., 1])

    add_colorbar(plot)


def find_config_file(dir):
    config_file = None
    for file in os.listdir(dir):
        if file.endswith(".gin"):
            config_file = os.path.join(dir, file)
    return config_file


def vis_mazes():
    maze_types = list(point_env_vis.ENV_LIMS.keys())

    fig, axes = plt.subplots(nrows=len(maze_types) // 4 + 1, ncols=4, figsize=(8, 8))

    for i in range(len(maze_types)):
        i1, i2 = i // 4, i % 4
        ax = axes[i1][i2]
        maze = maze_types[i]

        point_env_vis.config_subplot(ax, maze_type=maze)

    plt.show()


if __name__ == '__main__':
    vis_hierarchy_run("../logs/maze_square_bottleneck")