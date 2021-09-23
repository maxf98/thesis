import gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.environments import suite_gym

"""
env = gym.make('HalfCheetah-v2')
env.reset()

for _ in range(2000):
    env.render()
    print(env.step(env.action_space.sample()))
"""

#env = TFPyEnvironment(suite_gym.load("Hopper-v2"))
#pol = RandomTFPolicy(env.time_step_spec(), env.action_spec())
env = suite_gym.load("HalfCheetah-v2")
pol = RandomPyPolicy(env.time_step_spec(), env.action_spec())

ts = env.reset()

for _ in range(2000):
    env.render(mode='human')
    action_step = pol.action(ts)
    ts = env.step(action_step.action)
    if ts.is_last():
        print("terminated")

