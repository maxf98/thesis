import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import glob, os
from matplotlib import pyplot as plt

from env.maze import maze_env
from env.point_environment import PointEnv
from core.modules.policy_learners import SACLearner
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from tf_agents.utils.common import Checkpointer

"""
def buffer_size(rb):
    dataset = rb.as_dataset(single_deterministic_pass=True)
    mapped = list(dataset.map(lambda x, _: x.observation))
    return len(mapped)

checkpoint_dir = os.path.join(".", "resample_no_sn")

global_step = tf.compat.v1.train.get_or_create_global_step()

env = TFPyEnvironment(PointEnv())
agent = SACLearner(env.observation_spec(), env.action_spec(), env.time_step_spec(), reward_scale_factor=1.0)
buffer = TFUniformReplayBuffer(agent.policy.collect_data_spec, batch_size=env.batch_size)
driver = DynamicStepDriver(env, agent.policy, observers=[buffer.add_batch], num_steps=1000)
#driver.run()

#dataset = buffer.as_dataset(sample_batch_size=64, num_steps=2).prefetch(3)
#dataset_iter = iter(dataset)

#experience, unused_info = next(dataset_iter)
#agent.train(experience)


train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent.agent,
    policy=agent.agent.policy,
    replay_buffer=buffer,
    global_step=global_step
)

print(buffer_size(buffer))
train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_or_create_global_step()
print(buffer_size(buffer))


def annealed_entropy(step):
    entropy_anneal_period = 2500
    entropy_anneal_steps = 2000
    initial_entropy = 10
    target_entropy = 0.1

    step = step % entropy_anneal_period if entropy_anneal_period is not None else step

    alpha = initial_entropy - min((step / entropy_anneal_steps), 1) * (initial_entropy - target_entropy)

    return alpha

alphas = [annealed_entropy(i) for i in range(10000)]
rsfs = [1 / a for a in alphas]
fig, ax = plt.subplots()

ax.plot(range(10000), rsfs)

plt.show()
"""

# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)
sequential_model.save_weights("ckpt")
load_status = sequential_model.load_weights("ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
load_status.assert_consumed()