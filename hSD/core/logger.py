import os
from datetime import datetime
import matplotlib.pyplot as plt
from tf_agents.policies import policy_saver
import numpy as np
import shutil


class Logger:
    def __init__(self, log_dir, create_fig_interval):
        self.sac_stats = {'losses': []}
        self.discriminator_stats = {'losses': [], 'accuracy': []}

        date = datetime.now()
        dt_string = date.strftime("%d-%m~%H-%M")
        self.log_dir = os.path.join(log_dir, dt_string)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.copy_config_file()
        self.vis_dir = os.path.join(self.log_dir, "vis")
        os.mkdir(self.vis_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, discrim_stats, sac_stats):
        self.discriminator_stats['losses'].append(discrim_stats['losses'])
        self.discriminator_stats['accuracy'].append(discrim_stats['accuracy'])
        self.sac_stats['losses'].append(sac_stats['losses'])

        if epoch % self.create_fig_interval == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            dl = np.array(self.discriminator_stats["losses"]).flatten()
            da = np.array(self.discriminator_stats["accuracy"]).flatten()
            sl = np.array(self.sac_stats["losses"]).flatten()
            ax1.plot(range(len(da)), da, color='lightblue', linewidth=3)
            ax1.set(title='Discriminator accuracy')
            ax2.plot(range(len(dl)), dl, color='red', linewidth=3)
            ax2.set(title='Discriminator training loss')
            ax3.plot(range(len(sl)), sl, color='green', linewidth=3)
            ax3.set(title='SAC training loss')
            save_path = os.path.join(self.vis_dir, "epoch_{}".format(epoch))
            fig.savefig(save_path)
            plt.close(fig)


"""
class Logger:
    def __init__(self,
                 log_dir,
                 create_fig_interval):
        self.sac_stats = {'losses': []}
        self.discriminator_stats = {'losses': [], 'accuracy': []}

        date = datetime.now()
        dt_string = date.strftime("%d-%m~%H-%M")
        self.log_dir = os.path.join(log_dir, dt_string)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.copy_config_file()
        self.vis_dir = os.path.join(self.log_dir, "vis")
        os.mkdir(self.vis_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, policy, discriminator, env, num_skills, discrim_stats, sac_stats, exploration_rollouts):
        self.discriminator_stats['losses'].append(discrim_stats['losses'])
        self.discriminator_stats['accuracy'].append(discrim_stats['accuracy'])
        self.sac_stats['losses'].append(sac_stats['losses'])

        if epoch % self.create_fig_interval == 0:
            fig, ((ax1, ax2),(ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
            self.discrete_skill_vis(ax1, ax2, policy, discriminator, env, num_skills)
            # self.sample_policy_rollouts(ax1, policy, env, 70)
            dl = np.array(self.discriminator_stats["losses"]).flatten()
            da = np.array(self.discriminator_stats["accuracy"]).flatten()
            sl = np.array(self.sac_stats["losses"]).flatten()
            ax3.plot(range(len(da)), da, color='lightblue', linewidth=3)
            ax3.set(title='Discriminator accuracy')
            ax4.plot(range(len(dl)), dl, color='red', linewidth=3)
            ax4.set(title='Discriminator training loss')
            ax5.plot(range(len(sl)), sl, color='green', linewidth=3)
            ax5.set(title='SAC training loss')
            xs, ys = [p[0] for p in exploration_rollouts], [p[1] for p in exploration_rollouts]
            ax6.scatter(xs, ys, marker='.', alpha=0.2)
            ax6.set(title='Exploration map', xlim=[-1, 1], ylim=[-1, 1])
            save_path = os.path.join(self.vis_dir, "epoch_{}".format(epoch))
            fig.savefig(save_path)
            plt.close(fig)

    def close(self, policy, discriminator):
        # save policy and discriminator and produce loss figures
        self.save_discrim(discriminator)
        self.save_policy(policy)

    def copy_config_file(self):
        # uses relative path, not sure if that might cause some difficulties
        shutil.copy(os.path.abspath("../../configs/diayn_config.gin"), self.log_dir)
        # shutil.copy(os.path.abspath("configs/config.gin"), self.log_dir)

    @staticmethod
    def sample_policy_rollouts(ax, policy, env, path_length):
        point_env_vis.cont_skill_vis(ax, policy, env, path_length)

    @staticmethod
    def discrete_skill_vis(ax1, ax2, policy, discriminator, env, num_skills):
        point_env_vis.skill_vis(ax1, policy, num_skills, env, 5)
        point_env_vis.heatmap(ax2, discriminator)

    def save_discrim(self, discriminator):
        discriminator.save(os.path.join(self.log_dir, "discriminator"))

    def save_policy(self, policy):
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(os.path.join(self.log_dir, "policy"))
"""
