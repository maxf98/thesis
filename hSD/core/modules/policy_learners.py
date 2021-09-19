from abc import ABC, abstractmethod

import tensorflow as tf
from tf_agents.policies.tf_policy import TFPolicy
# from tf_agents.agents.sac import sac_agent
from thesis.hSD.lib import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.utils import common
from tf_agents.train.utils import train_utils


class PolicyLearner(ABC):
    def __init__(self, obs_spec, action_spec, time_step_spec,
                 initial_entropy, target_entropy, entropy_anneal_steps, entropy_anneal_period):
        """maximises rewards achieved by skill-conditioned policy"""
        self.obs_spec, self.action_spec, self.time_step_spec = obs_spec, action_spec, time_step_spec

        self.initial_entropy = initial_entropy
        self.target_entropy = target_entropy

        self.entropy_anneal_steps = entropy_anneal_steps
        # if not None, then we perform cyclical annealing, resetting to initial entropy every anneal_period steps
        self.entropy_anneal_period = entropy_anneal_period

    @abstractmethod
    def train(self, batch):
        """trains the policy"""

    @property
    def collect_policy(self):
        """returns current policy for collecting environment experience"""
        return None

    def anneal_alpha(self, agent):
        step = agent.train_step_counter.numpy()

        step = step % self.entropy_anneal_period if self.entropy_anneal_period is not None else step
        alpha = self.initial_entropy - min((step / self.entropy_anneal_steps), 1) * (self.initial_entropy - self.target_entropy)

        agent._reward_scale_factor = 1 / alpha

    @property
    def policy(self):
        """returns the current policy"""
        return None


class SACLearner(PolicyLearner):
    def __init__(self,
                 obs_spec,
                 action_spec,
                 time_step_spec,
                 network_fc_params=(128, 128),
                 initial_entropy=1.0,
                 target_entropy=None,
                 entropy_anneal_steps=10000,
                 entropy_anneal_period=None,
                 alpha_loss_weight=1.0
                 ):
        super(SACLearner, self).__init__(obs_spec, action_spec, time_step_spec,
                                         initial_entropy, target_entropy, entropy_anneal_steps, entropy_anneal_period)
        """
        initialise SAC rl agent and abstract some default hyperparameters
        """
        self.network_fc_params = network_fc_params
        self.optimizer = tf.keras.optimizers.Adam
        self.learning_rate = 3e-4
        self.target_update_tau = 0.005
        self.target_update_period = 1
        self.gamma = 1.0
        self.alpha_loss_weight = alpha_loss_weight
        self.reward_scale_factor = 1 / initial_entropy
        self.agent = self.initialise_sac_agent()

    def initialise_sac_agent(self):
        critic_net = critic_network.CriticNetwork(
            (self.obs_spec, self.action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=self.network_fc_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.obs_spec,
            self.action_spec,
            fc_layer_params=self.network_fc_params,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)

        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            self.time_step_spec,
            self.action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=self.optimizer(learning_rate=self.learning_rate),
            critic_optimizer=self.optimizer(learning_rate=self.learning_rate),
            alpha_optimizer=self.optimizer(learning_rate=self.learning_rate),
            alpha_loss_weight=self.alpha_loss_weight,
            target_update_tau=self.target_update_tau,
            target_update_period=self.target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=self.gamma,
            reward_scale_factor=self.reward_scale_factor,
            train_step_counter=train_step)

        tf_agent.initialize()

        tf_agent.train = common.function(tf_agent.train)

        return tf_agent

    def train(self, batch):
        sac_loss = self.agent.train(batch)

        if self.target_entropy is not None:
            self.anneal_alpha(self.agent)

        return sac_loss.loss

    @property
    def collect_policy(self):
        return self.agent.collect_policy

    @property
    def policy(self):
        return self.agent.policy

