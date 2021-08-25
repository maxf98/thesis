from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class PointEnv(py_environment.PyEnvironment):

  def __init__(self):
    step_size = 0.2
    self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-step_size, maximum=step_size, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='observation')
    self._state = (0., 0.)
    self._step_count = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0., 0.
    self._step_count = 0
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _clip_to_bounds(self, val, low, high):
    return min(high, max(val, low))

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    x_t, y_t = action.flatten()
    x = self._clip_to_bounds(self._state[0] + x_t, -1, 1)
    y = self._clip_to_bounds(self._state[1] + y_t, -1, 1)
    self._state = (x, y)

    self._step_count += 1

    if self._step_count > 100:
      self._episode_ended = True
      return ts.termination(np.array(self._state, dtype=np.float32), reward=0)
    else:
      return ts.transition(np.array(self._state, dtype=np.float32), reward=0)