import numpy as np
import tensorflow as tf
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

