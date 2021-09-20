import os
import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_agents.utils import common

from scripts import point_env_vis

from tf_agents.policies import policy_saver


#TODO: save optimizer state for models

@gin.configurable
class Logger:
    def __init__(self, log_dir, vis_skill_set, skill_length, create_fig_interval=1, num_samples_per_skill=1):
        self.sac_stats = {'loss': [], 'reward': []}
        self.skill_model_stats = {'loss': [], 'accuracy': []}

        self.log_dir = log_dir
        self.vis_dir = os.path.join(self.log_dir, "vis")
        self.policies_dir = os.path.join(self.log_dir, "policies")
        self.skill_weights_dir = os.path.join(self.log_dir, "skill_model_weights")
        self.stats_dir = os.path.join(self.log_dir, "stats")
        self.checkpoint_dir = os.path.join(self.log_dir, "tf_agent_checkpoints")

        self.vis_skill_set = vis_skill_set
        self.skill_length = skill_length
        self.num_samples_per_skill = num_samples_per_skill

        self.checkpointer = None

        self.create_fig_interval = create_fig_interval

    def initialize_or_restore(self, sd_agent):
        if not os.path.exists(self.log_dir):
            self.make_experiment_dirs()
            self.initialise_checkpointer(sd_agent.policy_learner.agent, sd_agent.rollout_driver.replay_buffer)
        else:
            self.initialise_checkpointer(sd_agent.policy_learner.agent, sd_agent.rollout_driver.replay_buffer)
            # only load things if there are things to load...
            #if tf.compat.v1.train.get_global_step() > 0:
            self.restore_skill_model_weights(sd_agent.skill_model)
            self.load_stats()
            tf.compat.v1.assign(tf.compat.v1.train.get_global_step(), 100)

    def make_experiment_dirs(self):
        os.makedirs(self.log_dir)
        os.mkdir(self.vis_dir)
        os.mkdir(self.policies_dir)
        os.mkdir(self.skill_weights_dir)
        os.mkdir(self.stats_dir)
        os.mkdir(self.checkpoint_dir)

    def log(self, epoch, skill_stats, sac_stats, timer_stats, policy_learner, skill_model, env):
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
            alpha = [policy_learner.alpha_for_step(i) for i in range(policy_learner.agent.train_step_counter.numpy())]

            ax1.plot(range(len(da)), da, color='lightblue', linewidth=3)
            ax1.set(title='Skill model accuracy')

            ax3.plot(range(len(dl)), dl, color='red', linewidth=3)
            ax3.set(title='Skill model training loss')

            ax2.plot(range(len(sl)), sl, color='red', linewidth=3)
            ax2.set(title='SAC training loss')

            ax4.plot(range(len(sr)), sr, color='green', linewidth=3)
            ax4.set(title='intrinsic reward')

            #point_env_vis.skill_vis(ax5, env, policy_learner.policy, self.vis_skill_set, self.num_samples_per_skill, self.skill_length)

            ax6.plot(range(len(alpha)), alpha, color='gray', linewidth=3)
            ax6.set(title='alpha')

            fig.set_size_inches(18.5, 10.5)
            fig.subplots_adjust(wspace=0.2, hspace=0.2)
            fig.suptitle("Epoch {}".format(epoch), fontsize=16)

            save_path = os.path.join(self.vis_dir, f"epoch_{epoch}")
            fig.savefig(save_path)
            plt.close(fig)

            # only update the global step when we actually save the agent state
            global_step = tf.compat.v1.train.get_global_step()
            tf.compat.v1.assign(global_step, epoch)

            if self.checkpointer is not None:
                self.checkpointer.save(global_step)

            self.save_policy(policy_learner.policy, epoch)
            self.save_skill_model_weights(skill_model)
            self.save_stats(timer_stats)

    def initialise_checkpointer(self, agent, replay_buffer):
        # also restores the global train step (epoch)... not the best way to do it, but it works
        train_step = tf.compat.v1.train.get_or_create_global_step()

        checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=train_step
        )

        self.checkpointer = checkpointer

        checkpointer.initialize_or_restore()

    def save_policy(self, policy, epoch):
        policy_dir = os.path.join(self.policies_dir, f"policy_{epoch}")
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(policy_dir)

    def save_skill_model_weights(self, skill_model):
        weights_dir = os.path.join(self.skill_weights_dir, f"weights")
        skill_model.model.save_weights(weights_dir)

    def restore_skill_model_weights(self, skill_model):
        """
        weights = sorted(os.listdir(self.skill_weights_dir))
        if len(weights) > 0:
            skill_model.model.load_weights(weights[-1])
        """
        skill_model.model.load_weights(os.path.join(self.skill_weights_dir, f"weights"))

    def load_stats(self):
        self.sac_stats['reward'] = np.load(os.path.join(self.stats_dir, "intrinsic_rewards.npy")).tolist()
        self.sac_stats['loss'] = np.load(os.path.join(self.stats_dir, "policy_loss.npy")).tolist()
        self.skill_model_stats['loss'] = np.load(os.path.join(self.stats_dir, "discrim_loss.npy")).tolist()
        self.skill_model_stats['accuracy'] = np.load(os.path.join(self.stats_dir, "discrim_acc.npy")).tolist()

    def save_stats(self, timer_stats):
        self.save_timer_stats(timer_stats)
        rewards = np.array(self.sac_stats['reward']).flatten()
        policy_loss = np.array(self.sac_stats['loss']).flatten()
        discrim_losses = np.array(self.skill_model_stats['loss']).flatten()
        discrim_acc = np.array(self.skill_model_stats['accuracy']).flatten()

        np.save(os.path.join(self.stats_dir, "intrinsic_rewards"), rewards)
        np.save(os.path.join(self.stats_dir, "policy_loss"), policy_loss)
        np.save(os.path.join(self.stats_dir, "discrim_loss"), discrim_losses)
        np.save(os.path.join(self.stats_dir, "discrim_acc"), discrim_acc)

    def save_timer_stats(self, stats):
        # at the moment just overwrites the previous stat to only contain the total train time
        filepath = os.path.join(self.stats_dir, 'timer_stats.txt')
        with open(filepath, 'w') as f:
            f.write(stats)