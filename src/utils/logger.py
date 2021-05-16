from env import point_env_vis
import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.rl_agent_losses = []
        self.discriminator_losses = []

        self.exploration_times = []
        self.discriminator_times = []
        self.rl_times = []

    def log(self, epoch, num_skills, policy, discriminator, env):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.figure(figsize=(7, 14))
        fig.suptitle("Epoch {}".format(epoch))
        point_env_vis.skill_vis(ax1, policy, num_skills, env, 10)
        point_env_vis.heatmap(ax2, discriminator)

        fig.savefig('logs/vis/epoch_{}'.format(epoch))
        plt.close(fig)


def log_dads(epoch, policy, discriminator, env):
    fig, ax = plt.subplots(1, 1)
    plt.figure(figsize=(7, 7))
    fig.suptitle("Epoch {}".format(epoch))
    point_env_vis.cont_skill_vis(ax, policy, env, 16)
    fig.savefig('logs/dads/vis/epoch_{}'.format(epoch))
    plt.close(fig)


def log(epoch, num_skills, policy, discriminator, env):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure(figsize=(7, 14))
    fig.suptitle("Epoch {}".format(epoch))
    point_env_vis.skill_vis(ax1, policy, num_skills, env, 10)
    point_env_vis.heatmap(ax2, discriminator)

    fig.savefig('logs/diayn/vis/epoch_{}'.format(epoch))
    plt.close(fig)