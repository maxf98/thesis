import gin
import tensorflow as tf
# from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.policies import policy_saver
import tempfile

import os
from diayn import diayn_agent
from diayn import diayn_discriminator
from utils import utils
from utils import logger
from env import point_environment

env_name = "LunarLanderContinuous-v2"  # @param {type:"string"}


@gin.configurable
def run_experiment(num_skills=4, replay_buffer_size=10000):
    # train_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    # eval_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    train_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    eval_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())

    obs_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))
    aug_obs_spec = utils.aug_obs_spec(obs_spec, num_skills)
    aug_ts_spec = utils.aug_time_step_spec(time_step_spec, num_skills)

    tf_agent = init_sac_agent(obs_spec=aug_obs_spec, action_spec=action_spec, ts_spec=aug_ts_spec)

    discriminator = diayn_discriminator.DIAYNDiscriminator(num_skills, intermediate_dim=128, input_shape=obs_spec.shape)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_size)

    logging = init_logging()

    skill_discovery = diayn_agent.DIAYNAgent(
        train_env=train_env,
        eval_env=eval_env,
        rl_agent=tf_agent,
        skill_discriminator=discriminator,
        replay_buffer=replay_buffer,
        logger=logging,
        num_skills=num_skills
    )

    train_skill_discovery(skill_discovery)

    tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    tf_policy_saver.save("logs/policy")

    discriminator.save("logs/discriminator")


@gin.configurable
def init_logging(log_dir, create_fig_interval):
    return logger.Logger(log_dir, create_fig_interval)


@gin.configurable
def train_skill_discovery(skill_discovery,
                          num_epochs=50,
                          initial_collect_steps=5000,
                          collect_steps_per_epoch=1000,  # turn into collect_episodes ?
                          dynamics_train_steps_per_epoch=32,
                          sac_train_steps_per_epoch=32):
    skill_discovery.train(num_epochs,
                          initial_collect_steps,
                          collect_steps_per_epoch,
                          dynamics_train_steps_per_epoch,
                          sac_train_steps_per_epoch)


@gin.configurable
def init_sac_agent(obs_spec,
                   action_spec,
                   ts_spec,
                   actor_fc_layer_params,
                   critic_joint_fc_layer_params,
                   learning_rate, target_update_tau,
                   target_update_period, gamma,
                   reward_scale_factor,
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


if __name__ == '__main__':
    gin.parse_config_file("configs/config.gin")
    run_experiment()
