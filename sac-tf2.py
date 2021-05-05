
import matplotlib.pyplot as plt
import tempfile

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment



tempdir = tempfile.gettempdir()

env_name = "LunarLanderContinuous-v2" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 50 # @param {type:"integer"}

initial_collect_steps = 1000 # @param {type:"integer"}
collect_steps_per_iteration = 1000 # @param {type:"integer"}
replay_buffer_max_length = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 200 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 1 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}

env = suite_pybullet.load(env_name)

collect_py_env = suite_pybullet.load(env_name)
collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
collect_env.reset()
eval_py_env = suite_pybullet.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
eval_env.reset()

observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))

train_step = train_utils.create_train_step()

tf_agent = sac_agent.SacAgent(
        time_step_spec,
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

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=collect_env.batch_size,
    max_length=replay_buffer_max_length)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=1024,
    num_steps=2).prefetch(3)
iterator = iter(dataset)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

random_policy = random_tf_policy.RandomTFPolicy(collect_env.time_step_spec(),
                                                collect_env.action_spec())
time_step = collect_env.reset()
for _ in range(1000):
    collect_env.render()
    action_step = random_policy.action(time_step)
    next_time_step = collect_env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)


def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(1):

    time_step = environment.reset()
    episode_return = 0.0

    for i in range(1000):
        environment.render()
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


returns = []

time_step = collect_env.reset()
for epoch in range(20):
    print("epoch {}".format(epoch + 1))
    for iteration in range(1000):
        if iteration % 100 == 0:
            print(iteration)
        collect_env.render()
        action_step = tf_agent.collect_policy.action(time_step)
        next_time_step = collect_env.step(action_step.action)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)

    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience).loss

    step = tf_agent.train_step_counter.numpy()
    print('step = {0}: loss = {1}'.format(step, train_loss))
    if (epoch + 1) % 5 == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
