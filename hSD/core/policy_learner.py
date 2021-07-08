from abc import ABC, abstractmethod
from tf_agents.policies.tf_policy import TFPolicy


class PolicyLearner(ABC):
    def __init__(self,
                 ):
        """maximises rewards achieved by skill-conditioned policy"""

    @abstractmethod
    def train(self, batch) -> TFPolicy:
        """trains the policy"""
