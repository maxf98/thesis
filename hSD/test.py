import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

"""
skill_prior = tfp.distributions.OneHotCategorical(logits=[1, 1, 1, 1], dtype=tf.float32)
env = TFPyEnvironment(PointEnv())
policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())
print(policy.collect_data_spec)
buffer = TFUniformReplayBuffer(policy.collect_data_spec, batch_size=env.batch_size)
driver = DynamicStepDriver(env, policy, observers=[buffer.add_batch], num_steps=1000)
driver.run()

dataset = buffer.as_dataset(single_deterministic_pass=True)
traj = dataset.map(lambda x, _: x)
print(list(traj))


shuffled = obs.shuffle(1000)

print(tf.stack(list(obs)))
print(tf.stack(list(obs)))
print(tf.stack(list(shuffled)))
"""

fig, ax = plt.subplots()

x = np.arange(10000).reshape((100, 100))
rand = np.random.randint(0, 4, (100, 100))
ax.imshow(rand, cmap=cm.tab10)
plt.show()