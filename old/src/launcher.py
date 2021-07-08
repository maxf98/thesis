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

from diayn import diayn_agent
from diayn import diayn_discriminator
from edld import edl_agent
from edld import vae_discriminator
from utils import utils
from utils import logger
from thesis.hSD.env import point_environment

env_name = "LunarLanderContinuous-v2"  # @param {type:"string"}


@gin.configurable
def run_experiment(latent_dim=2):
    # train_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    # eval_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    train_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    eval_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())

    obs_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))
    aug_obs_spec = utils.aug_obs_spec(obs_spec, latent_dim)
    aug_ts_spec = utils.aug_time_step_spec(time_step_spec, latent_dim)

    tf_agent = init_sac_agent(obs_spec=aug_obs_spec, action_spec=action_spec, ts_spec=aug_ts_spec)

    # 2 ==> obs_spec.shape (currently returns tensor shape [2])
    skill_discriminator = init_skill_discriminator(type=type, input_dim=2, latent_dim=latent_dim)

    replay_buffer = init_buffer(tf_agent.collect_data_spec, train_env.batch_size)

    logging = init_logging()

    skill_discovery = init_skill_discovery(type=type, train_env=train_env, eval_env=eval_env, agent=tf_agent,
                                           discriminator=skill_discriminator, buffer=replay_buffer, logger=logging,
                                           latent_dim=latent_dim)

    train_skill_discovery(skill_discovery)


@gin.configurable
def init_buffer(data_spec, batch_size, buffer_size=10000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec,
        batch_size=batch_size,
        max_length=buffer_size)


@gin.configurable
def init_skill_discriminator(type, input_dim, intermediate_dim, latent_dim):
    if type == "DIAYN":
        discriminator = diayn_discriminator.DIAYNDiscriminator(latent_dim, intermediate_dim, input_dim)
        return discriminator
    elif type == "EDL":
        discriminator = vae_discriminator.VAEDiscriminator(input_dim, intermediate_dim, latent_dim)
        return discriminator


@gin.configurable
def init_skill_discovery(type, train_env, eval_env, agent, discriminator, buffer, logger, latent_dim):
    if type == "DIAYN":
        skill_discovery = diayn_agent.DIAYNAgent(
            train_env=train_env,
            eval_env=eval_env,
            rl_agent=agent,
            skill_discriminator=discriminator,
            replay_buffer=buffer,
            logger=logger,
            num_skills=latent_dim
        )
        return skill_discovery
    elif type == "EDL":
        skill_discovery = edl_agent.EDLAgent(train_env, eval_env, discriminator, agent, buffer, logger, latent_dim)
        return skill_discovery


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
