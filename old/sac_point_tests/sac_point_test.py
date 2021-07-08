import tensorflow as tf
# from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils

from core import utils
from thesis.hSD.env import point_environment

env_name = "LunarLanderContinuous-v2"  # @param {type:"string"}

learning_rate = 3e-4
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

def init_goal_experiment():
    train_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    eval_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())

    obs_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))
    agent_dim = 2 * obs_spec.shape.as_list()[0]  # s, g
    agent_obs_spec = utils.aug_obs_spec(obs_spec, agent_dim)
    agent_ts_spec = utils.aug_time_step_spec(time_step_spec, agent_dim)

    tf_agent = init_sac_agent(obs_spec=agent_obs_spec, action_spec=action_spec, ts_spec=agent_ts_spec)

    """
    collect experience from environment - every state reached could serve as a goal state in itself
    """


def init_sac_agent(obs_spec,
                   action_spec,
                   ts_spec,
                   optimizer=tf.keras.optimizers.Adam):
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        obs_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)

    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        ts_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=optimizer(learning_rate=learning_rate),
        critic_optimizer=optimizer(learning_rate=learning_rate),
        alpha_optimizer=optimizer(learning_rate=learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

    tf_agent.initialize()

    tf_agent.train = common.function(tf_agent.train)

    return tf_agent