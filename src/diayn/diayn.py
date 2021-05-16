import tensorflow as tf
import tensorflow_probability as tfp

import utils

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

from diayn.diayn_discriminator import DIAYNDiscriminator
from utils import logger

initial_collect_steps = 3000
num_epochs = 100
collect_steps_per_epoch = 1000
train_steps_per_epoch = 100
train_batch_size = 128
eval_interval = 10


class SkillDiscoveryAlgorithm:
    def __init__(self,
                train_env: TFEnvironment,
                eval_env: TFEnvironment,
                rl_agent: TFAgent,
                discriminator: DIAYNDiscriminator,
                replay_buffer: ReplayBuffer,
                num_skills
               ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._rl_agent = rl_agent
        self._discriminator = discriminator
        self._replay_buffer = replay_buffer
        rl_dataset = self._replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=train_batch_size,
                num_steps=2).prefetch(3)
        discriminator_dataset = self._replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=train_batch_size,
                num_steps=2).prefetch(3)
        self._train_batch_iterator = iter(rl_dataset)
        self._discriminator_batch_iterator = iter(discriminator_dataset)

        self._num_skills = num_skills
        self._skill_prior = tfp.distributions.Categorical(probs=[1/num_skills for _ in range(num_skills)])

    def _seed_replay_buffer(self):
        self._rl_agent.train_step_counter.assign(0)
        random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                        self._train_env.action_spec())

        time_step = self._train_env.reset()
        self._step_env(time_step, random_policy, initial_collect_steps)

    def _rl_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return self._relabel_ir(experience)

    def _discriminator_training_batch(self):
        return next(self._discriminator_batch_iterator)

    def _relabel_ir(self, batch):
        obs_l, obs_r = tf.split(batch.observation, [1, 1], axis=1)
        obs_l, z = tf.split(obs_l, [-1, self._num_skills], axis=2)
        obs_l = tf.reshape(obs_l, [obs_l.shape[0], -1])
        z = tf.reshape(z, [obs_l.shape[0], -1])
        obs_r, _ = tf.split(obs_r, [-1, self._num_skills], axis=2)
        obs_r = tf.reshape(obs_r, [obs_l.shape[0], -1])

        obs_delta = tf.subtract(obs_r, obs_l)

        def new_rewards(x, y):
            pred_y = self._discriminator.call(x)

            probs = tf.reduce_max(tf.multiply(pred_y, y), axis=1)
            r = tf.subtract(tf.math.log(probs), tf.math.log(1/self._num_skills)) # add log(p(z)) term

            return r

        r = tf.stack([new_rewards(obs_delta, z), batch.reward[:,1]], axis=1)

        ret = batch.replace(reward=r)
        return ret

    def train(self):
        for epoch in range(1, num_epochs + 1):
            print('epoch {}'.format(epoch))
            time_step = self._train_env.reset()

            z = self._skill_prior.sample().numpy()
            for i in range(collect_steps_per_epoch if epoch != 1 else initial_collect_steps):
                if time_step.is_last():
                    z = self._skill_prior.sample().numpy()
                aug_time_step = utils.aug_time_step(time_step, z, self._num_skills)
                action_step = self._rl_agent.collect_policy.action(aug_time_step)
                next_time_step = self._train_env.step(action_step.action)
                aug_next_time_step = utils.aug_time_step(next_time_step, z, self._num_skills)
                traj = trajectory.from_transition(aug_time_step, action_step, aug_next_time_step)
                self._replay_buffer.add_batch(traj)

                time_step = next_time_step

            # update policy with rl_algorithm and train discriminator
            discrim_acc = 0.
            for _ in range(train_steps_per_epoch):
                experience, _ = self._discriminator_training_batch()
                t1, t2 = experience.observation[:, 0, :-self._num_skills], experience.observation[:, 1, :-self._num_skills]
                delta_o = tf.subtract(t2, t1)
                z = experience.observation[:, 0, -self._num_skills:]
                discrim_history = self._discriminator.train(delta_o, z)
                #discrim_acc += discrim_history.history["val_accuracy"][0]

                experience = self._rl_training_batch()
                sac_loss = self._rl_agent.train(experience)

            avg_discrim_acc = discrim_acc / train_steps_per_epoch
            step = self._rl_agent.train_step_counter.numpy()
            print('step = {0}: sac training loss = {1}'.format(step, sac_loss))
            print('step = {0}: discrim_val_acc = {1}'.format(step, avg_discrim_acc))

            if epoch % eval_interval == 0 or epoch == num_epochs:
                logger.log(epoch, self._num_skills, self._rl_agent.policy, self._discriminator, self._eval_env)

