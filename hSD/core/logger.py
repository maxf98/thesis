import os
from datetime import datetime
import matplotlib.pyplot as plt
from tf_agents.policies import policy_saver
import numpy as np
import tensorflow as tf
import shutil

from core.rollout_drivers import collect_skill_trajectories

from env import point_env_vis


class Logger:
    def __init__(self, log_dir, create_fig_interval, config_path, vis_skill_set):
        self.sac_stats = {'loss': [], 'reward': []}
        self.discriminator_stats = {'loss': [], 'accuracy': []}

        self.log_dir = log_dir

        self.vis_skill_set = vis_skill_set
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            date = datetime.now()
            dt_string = date.strftime("%d-%m~%H-%M")
            self.log_dir = os.path.join(log_dir, dt_string)
            os.mkdir(self.log_dir)

        self.copy_config_file(config_path)
        self.vis_dir = os.path.join(self.log_dir, "vis")
        os.mkdir(self.vis_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, discrim_stats, sac_stats, policy, discriminator, env, latent_dim):
        self.discriminator_stats['loss'].append(discrim_stats['loss'])
        self.discriminator_stats['accuracy'].append(discrim_stats['accuracy'])
        self.sac_stats['loss'].append(sac_stats['loss'])

        if epoch % self.create_fig_interval == 0:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
            dl = np.array(self.discriminator_stats["loss"]).flatten()
            da = np.array(self.discriminator_stats["accuracy"]).flatten()
            sl = np.array(self.sac_stats["loss"]).flatten()
            ax1.plot(range(len(da)), da, color='lightblue', linewidth=3)
            ax1.set(title='Discriminator accuracy')
            ax2.plot(range(len(dl)), dl, color='red', linewidth=3)
            ax2.set(title='Discriminator training loss')
            ax3.plot(range(len(sl)), sl, color='red', linewidth=3)
            ax3.set(title='SAC training loss')

            cur_policy_ir = self.cur_policy_eval_ir(env, policy, discriminator, 1, 30)
            self.sac_stats['reward'].append(cur_policy_ir)
            sr = np.array(self.sac_stats['reward']).flatten()
            ax4.plot(range(len(sr)), sr, color='green', linewidth=3)
            ax4.set(title='intrinsic reward')

            self.skill_vis(ax5, ax6, policy, discriminator, env)

            fig.set_size_inches(18.5, 10.5)
            fig.subplots_adjust(wspace=0.2, hspace=0.2)

            save_path = os.path.join(self.vis_dir, "epoch_{}".format(epoch))
            fig.savefig(save_path)
            plt.close(fig)

    def copy_config_file(self, config_path):
        # uses relative path, not sure if that might cause some difficulties
        shutil.copy(os.path.abspath(config_path), self.log_dir)
        # shutil.copy(os.path.abspath("configs/config.gin"), self.log_dir)

    def skill_vis(self, ax1, ax2, policy, discriminator, env):
        point_env_vis.skill_vis(ax1, env, policy, self.vis_skill_set, 1, 30)
        # point_env_vis.categorical_discrim_heatmap(ax2, discriminator)

    @staticmethod
    def cur_policy_eval_ir(env, policy, discriminator, rollouts_per_skill, skill_length):
        # ideally this would use the same rollout driver I suppose, but at the same time we're not training an RL agent from this,
        # so something that only return (s, z) pairs would be better, but for this simple case it#s fine...
        # this could be a method subsumed by the abstract base class
        # also, I guess technically we don't need to do this separately..
        num_skills = discriminator.latent_dim
        skills = tf.one_hot(list(range(num_skills)), num_skills)
        trajs = collect_skill_trajectories(env, policy, skills, rollouts_per_skill=rollouts_per_skill, trajectory_length=skill_length)
        trajs = tf.reshape(trajs, (num_skills, rollouts_per_skill * skill_length, discriminator.input_dim))
        log_probs = [discriminator.log_probs(trajs[i], skills[i]) for i in range(num_skills)]
        r = [tf.subtract(log_probs[i], tf.math.log(1 / num_skills)) for i in range(num_skills)]
        avg_reward = tf.reduce_mean(r)
        return avg_reward

    def save_stats(self):
        rewards = np.array(self.sac_stats['reward']).flatten()
        discrim_losses = np.array(self.discriminator_stats['loss']).flatten()
        discrim_acc = np.array(self.discriminator_stats['accuracy']).flatten()

        np.save(os.path.join(self.log_dir, "intrinsic_rewards"), rewards)
        np.save(os.path.join(self.log_dir, "discrim_loss"), discrim_losses)
        np.save(os.path.join(self.log_dir, "discrim_acc"), discrim_acc)

    def save_discrim(self, discriminator):
        discriminator.save(os.path.join(self.log_dir, "discriminator"))

    def save_policy(self, policy):
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(os.path.join(self.log_dir, "policy"))


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
