
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.specs import BoundedArraySpec

from core.modules import rollout_drivers

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
        """returns the tensor spec of the latent skill space, default is [-1, 1]^2"""
        return self._action_spec

    def observation_spec(self):
        return self._env.observation_spec()

    def _reset(self):
        return self._env.reset()

    def _step(self, action, return_intermediate_steps=False):
        time_step = self._env.current_time_step()

        time_steps = []

        time_step = rollout_drivers.rollout_skill_trajectory(time_step, self._env, self._policy, rollout_drivers.preprocess_time_step,
                                                 [time_steps.append], action, self._skill_length, return_aug_obs=False, s_norm=True)

        if return_intermediate_steps:
            return tf.reshape(time_steps, (-1))
        else:
            return time_step


