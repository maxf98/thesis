import tensorflow as tf
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.policies import policy_saver
import tempfile

import os
import skill_discovery_algorithm
import diayn_discriminator
import utils
import point_environment
import point_env_vis

env_name = "LunarLanderContinuous-v2" # @param {type:"string"}

num_skills = 4

sample_batch_size = 64
replay_buffer_max_length = 10000

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())


def run_experiment():
    # train_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    # eval_env = tf_py_environment.TFPyEnvironment(suite_pybullet.load(env_name))
    train_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())
    eval_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())

    obs_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(train_env))
    aug_obs_spec = utils.aug_obs_spec(obs_spec, num_skills)
    aug_ts_spec = utils.aug_time_step_spec(time_step_spec, num_skills)

    critic_net = critic_network.CriticNetwork(
        (aug_obs_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        aug_obs_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))

    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
            aug_ts_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

    tf_agent.initialize()

    tf_agent.train = common.function(tf_agent.train)

    discriminator = diayn_discriminator.DIAYNDiscriminator(obs_spec.shape, num_skills)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    skill_discovery = skill_discovery_algorithm.SkillDiscoveryAlgorithm(
        train_env=train_env,
        eval_env=eval_env,
        rl_agent=tf_agent,
        discriminator=discriminator,
        replay_buffer=replay_buffer,
        num_skills=num_skills
    )

    skill_discovery.train()

    policy_dir = os.path.join(tempdir, 'policy')
    tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
    tf_policy_saver.save(policy_dir)

    policy_zip_filename = utils.create_zip_file(policy_dir, os.path.join(tempdir, 'exported_policy'))
    utils.unzip_to(policy_zip_filename, "logs")

    point_env_vis.skill_vis(tf_agent.policy, num_skills, eval_env, 10)


if __name__ == '__main__':
    run_experiment()