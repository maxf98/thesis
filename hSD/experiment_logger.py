import os
import shutil
import numpy as np

from tf_agents.policies import policy_saver


class ExperimentLogger:
    def __init__(self,
                 log_dir, config_path):
        """this class is meant mostly for experiment saving, restoration, etc..."""
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            # technically this means that there is an existing experiment that we would like to continue...
            # this needs to be managed differently somehow, for now we assume there is no directory!
            print("directory exists already, you probably wanna handle that somehow...")

        self.copy_config_file(config_path)

    def copy_config_file(self, config_path):
        shutil.copy(config_path, self.log_dir)

    def get_layer_log_dir(self, layer):
        return os.path.join(self.log_dir, str(layer))

