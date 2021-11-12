import gym
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.environments import suite_gym
import tensorflow as tf

import launcher

from collections import OrderedDict
"""
env = gym.make('HalfCheetah-v2')
env.reset()

for _ in range(2000):
    env.render()
    print(env.step(env.action_space.sample()))
"""

env = launcher.get_base_env("handreach")
print(env.observation_spec(), env.action_spec())

