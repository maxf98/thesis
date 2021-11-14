import numpy as np
import tensorflow as tf
import itertools
import os
import numpy
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
    step = (max-min) / (points_per_axis - 1)
    skill_axes = [[min + step * x for x in range(points_per_axis)]] * dim
    skills = [skill for skill in itertools.product(*skill_axes)]
    return skills


def one_hots_for_num_skills(num_skills):
    return tf.one_hot(list(range(num_skills)), num_skills)


def points_along_axis(a, num_points, dim, default_value=0.0):
    skills = [[default_value for _ in range(dim)] for _ in range(num_points)]
    for p in range(num_points):
        skills[p][a] = -1. + p * (2 / num_points)
    return skills


def random_samples(min, max, dim, num_samples):
    return [[np.random.uniform(min, max) for _ in range(dim)] for _ in range(num_samples)]


def hide_goal(obs):
    return obs['observation'] if isinstance(obs, OrderedDict) else obs


def get_sorted_files(dir):
    return sorted(os.listdir(dir), key=lambda x: os.path.getmtime(os.path.join(dir, x)))


def graph_alpha(ax, steps, initial_entropy, target_entropy, entropy_anneal_steps, entropy_anneal_period, color="blue"):
    alphas = []
    for step in range(steps):
        step = step % entropy_anneal_period if entropy_anneal_period is not None else step
        alphas.append(initial_entropy - min((step / entropy_anneal_steps), 1) * (initial_entropy - target_entropy))
    ax.plot(range(steps), alphas, linewidth=2, color=color)

