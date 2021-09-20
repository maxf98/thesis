import gym

from tf_agents.environments import suite_gym
from tf_agents.policies.random_py_policy import RandomPyPolicy


#env = suite_gym.load("BipedalWalker-v3")
#rand_pol = RandomPyPolicy(env.time_step_spec(), env.action_spec())

env = gym.make("Hopper-v2")

env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())