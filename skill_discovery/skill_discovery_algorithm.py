import math

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import utils

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

from diayn_discriminator import DIAYNDiscriminator

initial_collect_steps = 1000
num_epochs = 50
collect_steps_per_epoch = 1000


class SkillDiscoveryAlgorithm():
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
                sample_batch_size=256,
                num_steps=2).prefetch(3)
        discriminator_dataset = self._replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=256,
                num_steps=1).prefetch(3)
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

    def _step_env(self, time_step, policy, num_steps):
        for i in range(num_steps):
            #self._train_env.render()
            action_step = policy.action(time_step)
            print(action_step.action)
            next_time_step = self._train_env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            self._replay_buffer.add_batch(traj)
            time_step = next_time_step

    def _rl_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return self._relabel_ir(experience)

    def _discriminator_training_batch(self):
        return next(self._discriminator_batch_iterator)

    def _relabel_ir(self, batch):
        obs_l, obs_r = tf.split(batch.observation, [1, 1], axis=1)

        def new_rewards(batch):
            x, y = tf.split(batch, [-1, self._num_skills], axis=2)
            y = tf.reshape(tf.argmax(y, axis=2), [-1])
            pred_y = self._discriminator.call(x)
            r = []
            for i in range(len(y)):
                ir = math.log(pred_y[i][y[i]])
                r.append(ir)

            return r

        r = tf.stack([new_rewards(obs_l), new_rewards(obs_r)], axis=1)

        ret = batch.replace(reward=r)
        return ret

    def train(self):
        for epoch in range(1, num_epochs + 1):
            print('epoch {}'.format(epoch))
            time_step = self._train_env.reset()

            z = self._skill_prior.sample().numpy()
            for i in range(collect_steps_per_epoch):

                aug_time_step = utils.aug_time_step(time_step, z, self._num_skills)
                action_step = self._rl_agent.collect_policy.action(aug_time_step)
                next_time_step = self._train_env.step(action_step.action)
                aug_next_time_step = utils.aug_time_step(next_time_step, z, self._num_skills)
                traj = trajectory.from_transition(aug_time_step, action_step, aug_next_time_step)
                self._replay_buffer.add_batch(traj)

                time_step = next_time_step

            # update policy with rl_algorithm
            for _ in range(100):
                experience = self._rl_training_batch()
                train_loss = self._rl_agent.train(experience)

            # train discriminator
            for _ in range(100):
                experience, _ = self._discriminator_training_batch()
                discrim_loss = self._discriminator.train(experience)

            step = self._rl_agent.train_step_counter.numpy()
            print('step = {0}: loss = {1}'.format(step, train_loss))
            print('step = {0}: val_acc = {1}'.format(step, discrim_loss.history["val_accuracy"]))



