import numpy as np
import tensorflow as tf
import itertools
import io
import shutil
import zipfile
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec


def aug_obs_spec(obs_spec, new_dim):
    return BoundedTensorSpec(shape=(new_dim, ), dtype=obs_spec.dtype,
                             name="augmented observation", minimum=obs_spec.minimum, maximum=obs_spec.maximum)


def aug_time_step_spec(time_step_spec: ts.TimeStep, new_dim):
    return ts.TimeStep(step_type=time_step_spec.step_type,
                       reward=time_step_spec.reward,
                       discount=time_step_spec.discount,
                       observation=aug_obs_spec(time_step_spec.observation, new_dim))


def aug_time_step(time_step, z):
    return ts.TimeStep(time_step.step_type,
                       time_step.reward,
                       time_step.discount,
                       tf.concat([time_step.observation, tf.reshape(z, (1, -1))], axis=-1))


def discretize_continuous_space(min, max, points_per_axis, dim):
    step = (max-min) / points_per_axis
    skill_axes = [[min + step * x for x in range(points_per_axis + 1)]] * dim
    skills = [skill for skill in itertools.product(*skill_axes)]
    return skills