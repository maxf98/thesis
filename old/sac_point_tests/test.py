from thesis.hSD.env import point_environment
from tf_agents.environments import tf_py_environment


train_env = tf_py_environment.TFPyEnvironment(point_environment.PointEnv())

time_step = train_env.reset()
x, y = time_step.observation.numpy().flatten()
print(x, y)