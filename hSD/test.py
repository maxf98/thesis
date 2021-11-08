import gym
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.environments import suite_gym
import tensorflow as tf

from collections import OrderedDict
"""
env = gym.make('HalfCheetah-v2')
env.reset()

for _ in range(2000):
    env.render()
    print(env.step(env.action_space.sample()))
"""


#env = TFPyEnvironment(suite_gym.load("Hopper-v2"))
#pol = RandomTFPolicy(env.time_step_spec(), env.action_spec())
env = suite_gym.load("FetchSlide-v1")
pol = RandomPyPolicy(env.time_step_spec(), env.action_spec())

print(env.observation_spec()['observation'])
ts_spec = env.time_step_spec()._replace(observation=env.observation_spec()['observation'])
print(ts_spec)
ts = env.reset()

"""
for _ in range(2000):
    env.render(mode='human')
    action_step = pol.action(ts)
    ts = env.step(action_step.action)
    if ts.is_last():
        print("terminated")

"""


t = [1, 2, 3, 4]
print(t[:-1])