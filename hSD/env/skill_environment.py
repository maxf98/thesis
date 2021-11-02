
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.specs import tensor_spec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from core.modules import utils

import numpy as np
import tensorflow as tf


class SkillEnv(py_environment.PyEnvironment):
    """wraps a python environment using a skill-conditioned policy,
    from which we choose skills to execute for T time-steps
    --- essentially replaces action space of py_env with skill_space"""
    def __init__(self,
                 env: py_environment.PyEnvironment,
                 policy: py_policy.PyPolicy,
                 skill_length,
                 skill_dim=None,
                 state_norm=True):
        super(SkillEnv, self).__init__()

        self._env = env
        self._policy = policy
        self._skill_length = skill_length
        self._action_spec = BoundedArraySpec(shape=(skill_dim,), dtype=np.float32, minimum=-1, maximum=1, name='action')

        self._state = self._env.reset()
        self._state_norm = state_norm

    def action_spec(self):
        """returns the tensor spec of the latent skill space, default is 2-dim (-1, 1)"""
        return self._action_spec

    def observation_spec(self):
        return self._env.observation_spec()

    def _reset(self):
        return self._env.reset()

    def _step(self, action, return_intermediate_steps=False):
        time_step = self._env.current_time_step()
        s_0 = time_step.observation
        time_steps = [s_0]

        for i in range(self._skill_length):
            aug_ts = self._preprocess_time_step(time_step, action, s_norm=s_0)
            action_step = self._policy.action(aug_ts)
            time_step = self._env.step(action_step.action)
            time_steps.append(time_step)

        if return_intermediate_steps:
            return tf.reshape(time_steps, [-1])
        else:
            return time_step

    def _preprocess_time_step(self, time_step, skill, s_norm):
        obs = utils.hide_goal(time_step.observation)
        obs = obs - s_norm if self._state_norm else obs
        return ts.TimeStep(time_step.step_type,
                           time_step.reward,
                           time_step.discount,
                           tf.concat([obs, skill], axis=-1))

