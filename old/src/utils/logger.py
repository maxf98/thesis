import os
from datetime import datetime
import matplotlib.pyplot as plt
from tf_agents.policies import policy_saver



class Logger:
    def __init__(self,
                 log_dir,
                 create_fig_interval):
        self.rl_agent_losses = []
        self.discriminator_losses = []

        self.exploration_times = []
        self.discriminator_times = []
        self.rl_times = []
        date = datetime.now()
        dt_string = date.strftime("%d-%m~%H-%M")
        self.log_dir = os.path.join(log_dir, dt_string)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.create_fig_interval = create_fig_interval

    def log(self, epoch, fig):
        save_path = os.path.join(self.log_dir, "epoch_{}".format(epoch))
        fig.savefig(save_path)
        plt.close(fig)

    def save_discrim(self, discriminator):
        discriminator.save(os.path.join(self.log_dir, "discriminator"))

    def save_policy(self, policy):
        tf_policy_saver = policy_saver.PolicySaver(policy)
        tf_policy_saver.save(os.path.join(self.log_dir, "policy"))
