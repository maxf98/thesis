import os, sys

import gym
from tf_agents.environments import suite_gym
from tf_agents.policies.random_py_policy import RandomPyPolicy

"""
env = suite_gym.load("CartPole-v1")
step = env.reset()
policy = RandomPyPolicy(env.time_step_spec(), env.action_spec())

for _ in range(200):
    env.render()
    action_step = policy.action(step)
    env.step(action_step.action)
"""

env = gym.make("Hopper-v2")
env.reset()

for _ in range(1500):
    env.render()
    env.step(env.action_space.sample())