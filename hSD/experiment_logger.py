import os
import shutil

from tf_agents.policies import policy_saver


class ExperimentLogger:
    def __init__(self,
                 log_dir, config_path):
        """this class is meant mostly for experiment saving, restoration, etc..."""
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            self.policies_dir = os.path.join(self.log_dir, "policies")
            os.makedirs(self.policies_dir)
        else:
            # technically this means that there is an existing experiment that we would like to continue...
            # this needs to be managed differently somehow, for now we assume not!
            pass

        self.copy_config_file(config_path)

    def copy_config_file(self, config_path):
        shutil.copy(config_path, self.log_dir)

    def get_layer_log_dir(self, layer):
        return os.path.join(self.log_dir, str(layer))

    def save_policy(self, policy, layer):
        policy_dir = os.path.join(self.policies_dir, f"policy_{layer}")
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(policy_dir)

    def save_discriminator_weights(self, weights, layer):
        weights_dir = os.path.join(self.log_dir, f"weights_{layer}")
        # need to implement saving and restoring here...
