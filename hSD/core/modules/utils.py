import numpy as np
import tensorflow as tf
import itertools
import io
import shutil
import zipfile
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec

from collections import OrderedDict


def aug_obs_spec(obs_spec, new_dim):
    return BoundedTensorSpec(shape=(new_dim, ), dtype=tf.float32,
                             name="augmented observation", minimum=obs_spec.minimum, maximum=obs_spec.maximum)


def discretize_continuous_space(min, max, points_per_axis, dim):
    step = (max-min) / points_per_axis
    skill_axes = [[min + step * x for x in range(points_per_axis + 1)]] * dim
    skills = [skill for skill in itertools.product(*skill_axes)]
    return skills


def one_hots_for_num_skills(num_skills):
    return tf.one_hot(list(range(num_skills)), num_skills)


def hide_goal(obs):
    return obs['observation'] if isinstance(obs, OrderedDict) else obs