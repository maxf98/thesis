import numpy as np
import tensorflow as tf
import io
import shutil
import zipfile
from tf_agents.trajectories import time_step as ts
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec


def concat_obs_z(obs, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = tf.one_hot([z], num_skills)
    return tf.concat([obs, z_one_hot], -1)


def split_aug_obs(aug_obs, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return obs, z


def aug_obs_spec(obs_spec, num_skills):
    return BoundedTensorSpec(shape=(obs_spec.shape[0] + num_skills,), dtype=obs_spec.dtype, name="augmented observation", minimum=obs_spec.minimum, maximum=obs_spec.maximum)


def aug_time_step_spec(time_step_spec: ts.TimeStep, num_skills):
    return ts.TimeStep(step_type=time_step_spec.step_type,
                              reward=time_step_spec.reward,
                              discount=time_step_spec.discount,
                              observation=aug_obs_spec(time_step_spec.observation, num_skills))


def aug_time_step(time_step, z, num_skills):
    return ts.TimeStep(time_step.step_type,
                       time_step.reward,
                       time_step.discount,
                       concat_obs_z(time_step.observation, z, num_skills))


def skill_for_one_hot(v):
    return tf.argmax(v, axis=0)


def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)


def unzip_to(zip_path, to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(to)




