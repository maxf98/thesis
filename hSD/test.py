import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from hSD.env.point_environment import PointEnv


env = TFPyEnvironment(PointEnv())
policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())
buffer = TFUniformReplayBuffer(policy.collect_data_spec, batch_size=env.batch_size)
driver = DynamicStepDriver(env, policy, observers=[buffer.add_batch], num_steps=1000)
driver.run()

dataset = buffer.as_dataset(single_deterministic_pass=True)
obs = dataset.map(lambda x, _: x.observation)
shuffled = obs.shuffle(1000)

print(tf.stack(list(obs)))
print(tf.stack(list(obs)))
print(tf.stack(list(shuffled)))