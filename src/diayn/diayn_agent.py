import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory

from diayn.diayn_discriminator import DIAYNDiscriminator
from skill_discovery import SkillDiscovery
from utils import utils

initial_collect_steps = 1000
num_epochs = 100
collect_steps_per_epoch = 1000
train_steps_per_epoch = 100
train_batch_size = 128
eval_interval = 1


class DIAYNAgent(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_discriminator: DIAYNDiscriminator,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 logger,
                 num_skills,
                 max_skill_length=30,
                 ):
        super().__init__(train_env, eval_env, skill_discriminator, rl_agent, replay_buffer, logger, max_skill_length)

        rl_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_batch_size,
            num_steps=2).prefetch(3)
        discriminator_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_batch_size,
            num_steps=2).prefetch(3)
        self._train_batch_iterator = iter(rl_dataset)
        self._discriminator_batch_iterator = iter(discriminator_dataset)

        self.num_skills = num_skills
        self._skill_prior = tfp.distributions.Categorical(probs=[1 / num_skills for _ in range(num_skills)])

    def _rl_training_batch(self):
        experience, _ = next(self._train_batch_iterator)
        return self._relabel_ir(experience)

    def _discriminator_training_batch(self):
        return next(self._discriminator_batch_iterator)

    def _relabel_ir(self, batch):
        obs_l, obs_r = tf.split(batch.observation, [1, 1], axis=1)
        obs_l, z = tf.split(obs_l, [-1, self.num_skills], axis=2)
        obs_l = tf.reshape(obs_l, [obs_l.shape[0], -1])
        z = tf.reshape(z, [obs_l.shape[0], -1])
        obs_r, _ = tf.split(obs_r, [-1, self.num_skills], axis=2)
        obs_r = tf.reshape(obs_r, [obs_l.shape[0], -1])

        obs_delta = tf.subtract(obs_r, obs_l)

        def new_rewards(x, y):
            pred_y = self.skill_discriminator.call(x)

            probs = tf.reduce_max(tf.multiply(pred_y, y), axis=1)
            r = tf.subtract(tf.math.log(probs), tf.math.log(1 / self.num_skills))  # add log(p(z)) term

            return r

        r = tf.stack([new_rewards(obs_delta, z), batch.reward[:, 1]], axis=1)

        ret = batch.replace(reward=r)
        return ret

    def _collect_env(self, steps):
        time_step = self.train_env.reset()
        z = self._skill_prior.sample().numpy()
        count = 0
        for step in range(steps):
            if time_step.is_last() or count > self.max_skill_length:
                z = self._skill_prior.sample().numpy()
                time_step = self.train_env.reset()
                count = 0

            aug_time_step = utils.aug_time_step(time_step, z, self.num_skills)
            action_step = self.rl_agent.collect_policy.action(aug_time_step)
            next_time_step = self.train_env.step(action_step.action)
            aug_next_time_step = utils.aug_time_step(next_time_step, z, self.num_skills)

            traj = trajectory.from_transition(aug_time_step, action_step, aug_next_time_step)
            self.replay_buffer.add_batch(traj)

            time_step = next_time_step
            count += 1

    def _train_discriminator(self, steps):
        discrim_acc = 0

        for _ in range(steps):
            experience, _ = self._discriminator_training_batch()
            t1, t2 = experience.observation[:, 0, :-self.num_skills], experience.observation[:, 1, :-self.num_skills]
            delta_o = tf.subtract(t2, t1)
            z = experience.observation[:, 0, -self.num_skills:]
            discrim_history = self.skill_discriminator.train(delta_o, z)
            #discrim_acc += discrim_history["accuracy"][0]

        step = self.rl_agent.train_step_counter.numpy()
        #print('step = {0}: discrim_val_acc = {1}'.format(step, discrim_acc / steps))

    def _train_agent(self, steps):
        for _ in range(steps):
            experience = self._rl_training_batch()
            sac_loss = self.rl_agent.train(experience)

        step = self.rl_agent.train_step_counter.numpy()
        print('step = {0}: sac training loss = {1}'.format(step, sac_loss))

    def _log_epoch(self, epoch):
        if epoch % eval_interval == 0 or epoch == num_epochs:
            self.logger.log(epoch, self.num_skills, self.rl_agent.policy, self.skill_discriminator, self.eval_env)
