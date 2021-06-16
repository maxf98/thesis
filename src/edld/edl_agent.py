from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories import trajectory

from edld.vae_discriminator import VAEDiscriminator
from skill_discovery import SkillDiscovery
from utils import utils
import matplotlib.pyplot as plt
from env import point_env_vis

"""
considerations:
 - do we want to gather experiences in the environment with the skill-conditioned policy? 
    this might lead to more temporally extended behaviours while collecting, but also possibly lead to vicious cycle?
 - for now, we collect - encode - embed in one iteration, with long stages => easier to analyze!
 
 - I like the simple difference of not training the VAE with possibly faulty skill labels!

"""
class EDLAgent(SkillDiscovery):
    def __init__(self,
                 train_env: TFEnvironment,
                 eval_env: TFEnvironment,
                 skill_discriminator: VAEDiscriminator,
                 rl_agent: TFAgent,
                 replay_buffer: ReplayBuffer,
                 logger,
                 latent_dim,
                 max_skill_length=30,
                 train_batch_size=128
                 ):
        super().__init__(train_env, eval_env, skill_discriminator, rl_agent, replay_buffer, logger, max_skill_length)

        # exploration with random policy
        self.exploration_policy = RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        # just overwrite the wrong (skill-conditioned) replay buffer for now
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(self.exploration_policy.collect_data_spec, self.train_env.batch_size, 100000)

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

        self._train_batch_size = train_batch_size

        self.latent_dim = latent_dim
        self._skill_prior = tfp.distributions.Uniform(low=[-1.0, -1.0], high=[1.0, 1.0])

    """
    collect unlabelled experiences from environment, aim is just to achieve a good coverage of state space
    we use a skill-conditioned policy to gather, but don't assume ground truth in experiences
    that actually makes interleaving skills make more sense too!
    
    --> for now we just collect unlabelled experiences with a non-skill-conditioned exploration policy
    initially simply a random one
    
    we are collecting single steps, but depending on what we encode, this may change to trajectories
    """
    def _collect_env(self, steps):
        time_step = self.train_env.reset()

        for _ in tqdm(range(steps)):
            action_step = self.exploration_policy.action(time_step)
            next_time_step = self.train_env.step(action_step.action)

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            self.replay_buffer.add_batch(traj)

            time_step = next_time_step

    def _discriminator_training_batch(self):
        experience, _ = next(self._discriminator_batch_iterator)
        return experience

    """
    use beta-VAE? 
    discriminator expects unaugmented experiences --> for now state-differences
    encoder: delta_s -> z
    decoder: z -> delta_s
    """
    def _train_discriminator(self, steps):
        discrim_acc = 0

        for _ in tqdm(range(steps)):
            experience = self._discriminator_training_batch()
            t1, t2 = experience.observation[:, 0], experience.observation[:, 1]
            delta_o = tf.subtract(t2, t1)
            discrim_history = self.skill_discriminator.train(delta_o)
            # discrim_acc += discrim_history["accuracy"][0]

        step = self.rl_agent.train_step_counter.numpy()
        self.logger.save_discrim(self.skill_discriminator)

    def inspect_decoder(self):
        skill_prior = tfp.distributions.Uniform(low=[-1., -1.], high=[1., 1.])
        z = skill_prior.sample(10)
        delta_s = self.skill_discriminator.decode(z)
        print(delta_s)

    """
    relabel ir by sampling z and checking how well transition fits it
    --> data augmentation! each experience can be labelled multiple times with different skills
    ooor perform on-policy training...
    what we want is just a policy that can achieve the desired state difference in the MDP
    using SAC seems like overkill... but we'll try it for now! 
    why do the other approaches not use data augmentation.. seems like a simple way to possibly improve performance?
    """
    def _relabel_ir(self, batch):
        # update reward to reflect intrinsic reward, and concatenate observation with skill
        # importance sampling? not sure what it is...

        t1, t2 = batch.observation[:, 0, :], batch.observation[:, 1, :]
        delta_o = tf.subtract(t2, t1)

        # data augmentation
        num_prior_samples = 3
        relabelled_batch = None

        for _ in range(num_prior_samples):
            z = self._skill_prior.sample(self._train_batch_size)
            log_probs = self.skill_discriminator.log_probs(delta_o, z)  # add log p(z) term? makes sense here? nah...
            new_reward = tf.stack([log_probs, batch.reward[:, 1]], axis=1)
            new_batch = trajectory.Trajectory(step_type=batch.step_type, observation=[], action=batch.action, policy_info=batch.policy_info,
                                              next_step_type=batch.next_step_type, reward=new_reward, discount=batch.discount)

            aug_obs1 = tf.concat([batch.observation[:, 0, :], z], axis=1)
            aug_obs2 = tf.concat([batch.observation[:, 1, :], z], axis=1)
            aug_obs = tf.stack([aug_obs1, aug_obs2], axis=1)
            new_batch = new_batch.replace(observation=aug_obs)
            if relabelled_batch is None:
                relabelled_batch = new_batch
            else:
                relabelled_batch = utils.concat_trajectory(relabelled_batch, new_batch)


        return relabelled_batch


    def _rl_training_batch(self):
        batch, _ = next(self._train_batch_iterator)
        batch = self._relabel_ir(batch)
        return batch

    """
    agent embeds discriminator in MDP
    learns skill-conditioned policy so we can replace action space of next level of hierarchy with current latent space
    want: a policy that can achieve the skill difference encoded in the respective skill
    i feel like there should be a more direct way to encode that in a policy, than through off-policy, relabelled samples and SAC
    ==> world models? just use a simple neural network / linear function approximation
     to learn to choose an action such that reward is maximised (because credit assignment performed with world model)
    
    entropy term improves robustness of SAC policy... mimic? shouldn't the VAE take care of that? --> beta-vae?
    if I don't use SAC, may pave the way for discrete skills rather than continuous latent skill space...
    preferable for analysis, understandability and comparability
    ==> we try this IM approach to compare with later, but I think it doesn't make much sense...
    """
    def _train_agent(self, steps):
        sac_losses = []
        for i in tqdm(range(steps)):
            batch = self._rl_training_batch()
            loss_info = self.rl_agent.train(batch)
            sac_losses.append(loss_info)

        return sac_losses

    def _log_epoch(self, epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.figure(figsize=(7, 14))
        fig.suptitle("Epoch {}".format(epoch))
        point_env_vis.cont_skill_vis(ax1, self.rl_agent.policy, self.eval_env, 10)
        point_env_vis.latent_skill_vis(ax2, self.skill_discriminator.decoder)

        self.logger.log(epoch, fig)
