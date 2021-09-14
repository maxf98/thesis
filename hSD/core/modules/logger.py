import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from tf_agents.utils import common

from env import point_env_vis

from tf_agents.policies import policy_saver

#TODO: make train statistics depend on global step, not data length... this way we can pick up where we left off!

class Logger:
    def __init__(self, log_dir, create_fig_interval, vis_skill_set, skill_length, num_samples_per_skill):
        self.sac_stats = {'loss': [], 'reward': []}
        self.skill_model_stats = {'loss': [], 'accuracy': []}

        self.log_dir = log_dir

        self.vis_skill_set = vis_skill_set
        self.skill_length = skill_length
        self.num_samples_per_skill = num_samples_per_skill

        self.checkpointer = None

        # creates a directory inside the existing one, not a great way to handle this...
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            date = datetime.now()
            dt_string = date.strftime("%d-%m~%H-%M")
            self.log_dir = os.path.join(log_dir, dt_string)
            os.mkdir(self.log_dir)

        self.vis_dir = os.path.join(self.log_dir, "vis")
        os.mkdir(self.vis_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, global_step, skill_stats, sac_stats, policy, skill_model, env):
        if self.checkpointer is not None:
            self.checkpointer.save(global_step)

        self.skill_model_stats['loss'].append(skill_stats['loss'])
        self.skill_model_stats['accuracy'].append(skill_stats['accuracy'])
        self.sac_stats['loss'].append(sac_stats['loss'])
        self.sac_stats['reward'].append(sac_stats['reward'])

        if epoch % self.create_fig_interval == 0:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
            dl = np.array(self.skill_model_stats["loss"]).flatten()
            da = np.array(self.skill_model_stats["accuracy"]).flatten()
            sl = np.array(self.sac_stats["loss"]).flatten()
            sr = np.array(self.sac_stats['reward']).flatten()

            ax1.plot(range(len(da)), da, color='lightblue', linewidth=3)
            ax1.set(title='Skill model accuracy')

            ax2.plot(range(len(dl)), dl, color='red', linewidth=3)
            ax2.set(title='Skill model training loss')

            ax3.plot(range(len(sl)), sl, color='red', linewidth=3)
            ax3.set(title='SAC training loss')

            ax4.plot(range(len(sr)), sr, color='green', linewidth=3)
            ax4.set(title='intrinsic reward')

            self.skill_vis(ax5, ax6, policy, skill_model, env)

            fig.set_size_inches(18.5, 10.5)
            fig.subplots_adjust(wspace=0.2, hspace=0.2)
            fig.suptitle("Epoch {}".format(epoch), fontsize=16)

            save_path = os.path.join(self.vis_dir, f"epoch_{epoch}")
            fig.savefig(save_path)
            plt.close(fig)

            self.save_policy(policy, epoch)

    def skill_vis(self, ax1, ax2, policy, skill_model, env):
        point_env_vis.skill_vis(ax1, env, policy, self.vis_skill_set, self.num_samples_per_skill, self.skill_length)
        #point_env_vis.categorical_discrim_heatmap(ax2, skill_model)
        #point_env_vis.cont_diayn_skill_heatmap(ax2, skill_model)

    def per_skill_collect_rollouts(self, epoch, collect_policy, env):
        if epoch % self.create_fig_interval != 0:
            return

        # it would be nice if they weren't all in one row, if we did choose more to visualise
        fig, axes = plt.subplots(nrows=len(self.vis_skill_set) // 4 + 1, ncols=4, figsize=(8, 8))
        for i, skill in enumerate(self.vis_skill_set):
            point_env_vis.skill_vis(axes[i], env, collect_policy, [skill], 10, self.skill_length)
            axes[i].set(title=skill)
            axes[i].set_aspect('equal', adjustable='box')

        fig.suptitle("Epoch {}".format(epoch), fontsize=16)
        save_path = os.path.join(self.vis_dir, "epoch_{} - exploration rollouts".format(epoch))
        fig.savefig(save_path)
        plt.close(fig)

    def save_stats(self):
        rewards = np.array(self.sac_stats['reward']).flatten()
        discrim_losses = np.array(self.skill_model_stats['loss']).flatten()
        discrim_acc = np.array(self.skill_model_stats['accuracy']).flatten()

        np.save(os.path.join(self.vis_dir, "intrinsic_rewards"), rewards)
        np.save(os.path.join(self.vis_dir, "discrim_loss"), discrim_losses)
        np.save(os.path.join(self.vis_dir, "discrim_acc"), discrim_acc)

    def initialise_checkpointer(self, agent, replay_buffer, train_step, skill_model):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=train_step
        )

        self.checkpointer = checkpointer
        checkpointer.initialize_or_restore()

    def save_policy(self, policy, epoch):
        policy_dir = os.path.join(self.log_dir, f"policy_{epoch}")
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(policy_dir)