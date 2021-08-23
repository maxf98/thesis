import tensorflow as tf
from tqdm import tqdm

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory

from skill_discovery import SkillDiscovery
from core.discriminator import Discriminator
from core.modules import utils, logger as lg


class GoalConditionedSkillDiscovery(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_discriminator: Discriminator,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 logger: lg.Logger,
                 skill_prior,
                 latent_dim,
                 max_skill_length,
                 use_state_delta=False,
                 train_batch_size=128
                 ):
        super().__init__(train_env, eval_env, skill_discriminator, rl_agent, replay_buffer, logger, max_skill_length)

        buffer_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_batch_size,
            num_steps=2).prefetch(3)
        self._train_batch_iterator = iter(buffer_dataset)

        self.latent_dim = latent_dim
        self._skill_prior = skill_prior
        self._use_state_delta = use_state_delta

        self.exploration_rollouts = []

    def _rl_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return self._relabel_ir(experience)

    def _discriminator_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return experience

    def _collect_env(self, steps):
        """
        collect trajectories from skill-conditioned policy and relabel to concatenate with final state
        how often do we change the skill?
        """

        time_step = self.train_env.reset()
        z = self._skill_prior.sample()
        time_steps = []
        actions = []
        for step in tqdm(range(steps)):
            if time_step.is_last() or len(time_steps) == self.max_skill_length:
                # relabel experiences in temporary buffer and sample a new skill
                gz = tf.concat([time_steps[-1].observation, tf.reshape(z, [1, -1])], axis=1)
                cur_traj = [utils.aug_time_step(ts, gz) for ts in time_steps]
                for i in range(len(cur_traj) - 1):
                    traj = trajectory.from_transition(cur_traj[i], actions[i], cur_traj[i+1])
                    self.replay_buffer.add_batch(traj)

                time_steps, actions = [], []
                z = self._skill_prior.sample()

                time_step = self.train_env.reset()  # for now, skills radiate out from start!

            self.exploration_rollouts.append(time_step.observation.numpy().flatten())
            aug_time_step = utils.aug_time_step(time_step, tf.reshape(z, [1, -1]))
            action_step = self.rl_agent.collect_policy.action(aug_time_step)
            next_time_step = self.train_env.step(action_step.action)

            time_steps.append(time_step)
            actions.append(action_step)

            time_step = next_time_step

    def _train_discriminator(self):
        batch = self._discriminator_training_batch()
        x = batch.observation[:, 0, :-self.latent_dim]
        y = batch.observation[:, 0, -self.latent_dim:]
        discrim_history = self.skill_discriminator.train(x, y)
        return discrim_history.history['loss'], discrim_history.history['accuracy']

    def _relabel_ir(self, batch):
        # relabel with intrinsic reward and perform data augmentation
        # should return batch in which observations are concatenated state and skill, i.e. tensor spec CHANGES
        # batch is a trajectory object with obs_dim (batch_size, 2, 2 * obs_dim + latent_dim)
        batch_size = batch.observation.shape.as_list()[0]
        obs_l, obs_r = tf.split(batch.observation, [1, 1], axis=1)
        obs_l, obs_r = tf.reshape(obs_l, (batch_size, -1)), tf.reshape(obs_r, (batch_size, -1))
        env_obs_dim = self.train_env.observation_spec().shape.as_list()[0]
        s_l, g_l, z_l = tf.split(obs_l, [env_obs_dim, env_obs_dim, -1], axis=1)
        s_r, g_r, z_r = tf.split(obs_r, [env_obs_dim, env_obs_dim, -1], axis=1)

        # not sure whether SAC uses reward of second time step for training, so not sure whether need to compute it,
        # but we'll just try it out
        x_l, x_r = tf.concat([s_l, g_l], axis=1), tf.concat([s_r, g_r], axis=1)
        new_r = tf.stack([self.skill_discriminator.log_prob(x_l, z_l),
                          self.skill_discriminator.log_prob(x_r, z_r)], axis=1)

        new_obs = tf.stack([tf.concat([s_l, z_l], axis=1),
                            tf.concat([s_r, z_r], axis=1)], axis=1)

        ret = batch.replace(reward=new_r)
        ret = ret.replace(observation=new_obs)

        return ret

    def _train_agent(self):
        experience = self._rl_training_batch()
        sac_loss = self.rl_agent.train(experience)

        return sac_loss.loss.numpy()

    def _log_epoch(self, epoch, discrim_stats, sac_stats):
        """
        log sac and discriminator losses and create a figure sampling rollouts of current skill-conditioned policy in environment
        """
        self.logger.log(epoch, self.rl_agent.policy, self.skill_discriminator, self.eval_env, self.latent_dim,
                        discrim_stats, sac_stats, self.exploration_rollouts)
        self.exploration_rollouts = []
