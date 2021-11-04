
import tensorflow as tf
import numpy as np
#from env.mazes import mazes_dict, make_crazy_maze, make_experiment_maze, make_hallway_maze, make_u_maze
from env.maze.mazes import mazes_dict, make_crazy_maze, make_experiment_maze, make_hallway_maze, make_u_maze

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

class MazeEnv(PyEnvironment):
    def __init__(self, maze_type=None, action_range=None):
        super(PyEnvironment, self).__init__()

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}

        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes

        self.maze = self._mazes[self.maze_type]['maze']
        if action_range is None:  # if unspecified, use default action range given by EDL authors
            action_range = self._mazes[self.maze_type]['action_range']

        self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-action_range,
                                                        maximum=action_range, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, name='observation')

        self._state = self.reset()

        self._maximum_episode_length = 1000
        self._episode_ended = False
        self._step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._step_count = 0
        self._state = self.maze.sample_start()  # for training this is fine, for visualisation we want to fix the start
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        # at the moment we arbitrarily terminate episodes after 1000 steps
        # but only to allow for tf_agents validate_py_env method to work
        if self._episode_ended:
            self.reset()

        try:
            next_state = self.maze.move(self._state, action)
        except:
            print('state', self.to_coords(self._state))
            print('action', action)
            raise

        self._state = next_state
        self._step_count += 1

        if self._step_count > 1000:
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.float32), reward=0)

        return ts.transition(np.array(self._state, dtype=np.float32), reward=0)

    def plot(self):
        self.maze.plot()

