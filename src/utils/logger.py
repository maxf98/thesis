import os
from datetime import datetime
from env import point_env_vis
import matplotlib.pyplot as plt


class Logger:
    def __init__(self,
                 log_dir,
                 create_fig_interval):
        self.rl_agent_losses = []
        self.discriminator_losses = []

        self.exploration_times = []
        self.discriminator_times = []
        self.rl_times = []
        date = datetime.now()
        dt_string = date.strftime("%d-%m~%H-%m")
        self.log_dir = os.path.join(log_dir, dt_string)
        os.mkdir(self.log_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, num_skills, policy, discriminator, env):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.figure(figsize=(7, 14))
        fig.suptitle("Epoch {}".format(epoch))
        point_env_vis.skill_vis(ax1, policy, num_skills, env, 10)
        point_env_vis.heatmap(ax2, discriminator)

        save_path = os.path.join(self.log_dir, "epoch_{}".format(epoch))
        fig.savefig(save_path)
        plt.close(fig)

    def create_figures(self):
        # TODO
        # create figures, either every time something is logged, or after experiment finished running
        pass


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