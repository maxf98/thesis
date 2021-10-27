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

  def __init__(self, step_size=0.1, box_size=None, dim=3):
    """we keep the box_size parameter for now, to not break configs that use it..."""
    super(PointEnv, self).__init__()
    assert(step_size > 0)

    self.step_size = step_size
    self.start_state = tuple(0. for _ in range(dim))
    self._action_spec = array_spec.BoundedArraySpec(shape=(dim,), dtype=np.float32, minimum=-step_size, maximum=step_size, name='action')
    self._observation_spec = array_spec.ArraySpec(shape=(dim,), dtype=np.float32, name='observation')
    self._state = self.start_state
    self._step_count = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = self.start_state
    self._step_count = 0
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.float32))

  @staticmethod
  def _clip_to_bounds(val, low, high):
    return min(high, max(val, low))

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    """
    x_t, y_t = action.flatten()
    x = self._clip_to_bounds(self._state[0] + x_t, -self.box_size, self.box_size)
    y = self._clip_to_bounds(self._state[1] + y_t, -self.box_size, self.box_size)
    self._state = (x, y)
    """

    self._state = tuple(self._state[i] + action[i] for i in range(len(self._state)))

    self._step_count += 1

    if self._step_count > 1000:
      self._episode_ended = True
      return ts.termination(np.array(self._state, dtype=np.float32), reward=0)
    else:
      return ts.transition(np.array(self._state, dtype=np.float32), reward=0)

  def set_start_state(self, state):
    self.start_state = state
    self.reset()