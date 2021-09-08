import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory

from core import skill_discovery
import skill_dynamics
from utils import logger
from utils import utils

batch_size = 128


class DADS(skill_discovery.SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_dynamics: skill_dynamics.SkillDynamics,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 max_skill_length,
                 ):
        super().__init__(train_env, eval_env, skill_dynamics, rl_agent, replay_buffer, max_skill_length)
        self.skill_discriminator = skill_dynamics

        # agent and discriminator both need timesteps as (s, a, s', r)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self.train_batch_iterator = iter(dataset)

        self.skill_prior = tfp.distributions.Uniform([-1, -1], [1, 1])

    # assumes that a skill is kept constant for an entire trajectory!
    # could reimplement to collect specific number of episodes, rather than steps...
    def _collect_env(self, steps):
        time_step = self.train_env.reset()
        z = self.skill_prior.sample()
        count = 0
        for step in range(steps):
            if time_step.is_last() or count > self.max_skill_length:
                z = self.skill_prior.sample()
                time_step = self.train_env.reset()
                count = 0

            aug_time_step = utils.aug_time_step_cont(time_step, z)
            action_step = self.rl_agent.collect_policy.action(aug_time_step)
            next_time_step = self.train_env.step(action_step.action)
            aug_next_time_step = utils.aug_time_step_cont(next_time_step, z)

            traj = trajectory.from_transition(aug_time_step, action_step, aug_next_time_step)
            self.replay_buffer.add_batch(traj)

            time_step = next_time_step
            count += 1


    def _train_discriminator(self, steps):
        print("Start discriminator training")
        for i in range(steps):
            print("iteration {}".format(i + 1))
            train_batch, _ = next(self.train_batch_iterator)
            batch = train_batch.observation
            x = batch[:, 0, :]
            # compute state delta
            y = tf.subtract(batch[:, 1, :-self.skill_discriminator.latent_dim], batch[:, 0, :-self.skill_discriminator.latent_dim])
            self.skill_discriminator.train(x, y)

    def _train_agent(self, steps):
        print("Start SAC training")
        for i in range(steps):
            print("iteration {}".format(i + 1))
            experience, _ = next(self.train_batch_iterator)
            train_batch = self._relabel_ir(experience)
            self.rl_agent.train(train_batch)

    def _relabel_ir(self, traj: trajectory.Trajectory):
        """ relabel transitions with intrinsic reward
        can only relabel first reward... reward at next step? impact? I guess not considered in SAC anyway...)"""
        # get Trajectory of transitions w/ obs 128x2x4 batch, compute reward with skill_dynamics...
        # need to compute likelihood of actual next state under computed dynamics model
        r = self.skill_discriminator.log_prob(traj.observation)
        new_rewards = tf.stack([r, traj.reward[:, 1]], axis=1)

        return traj.replace(reward=new_rewards)

    def _log_epoch(self, epoch):
        logger.log_dads(epoch, self.rl_agent.policy, self.skill_discriminator, self.eval_env)


