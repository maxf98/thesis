from typing import Optional

from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types

import utils


# wrapper class for policy to fix skill
class FixedOptionPolicy(tf_policy.TFPolicy):
    def __init__(self, policy, skill, num_skills):
        self._policy = policy
        self._skill = skill
        self._num_skills = num_skills

    # expects unaugmented observations and augments them for use with parent policy
    def action(self, time_step: ts.TimeStep, policy_state: types.NestedTensor = (), seed: Optional[types.Seed] = None):
        aug_ts = utils.aug_time_step(time_step, self._skill, self._num_skills)
        return self._policy.action(aug_ts)

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        aug_ts = utils.aug_time_step(time_step, self._skill, self._num_skills)
        return self._policy.distribution(aug_ts, policy_state)


class FixedOptionPolicyCont(tf_policy.TFPolicy):
    def __init__(self, policy, skill):
        self.policy = policy
        self.skill = skill

    def action(self, time_step: ts.TimeStep, policy_state: types.NestedTensor = (), seed: Optional[types.Seed] = None):
        aug_ts = utils.aug_time_step_cont(time_step, self.skill)
        return self.policy.action(aug_ts)

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        aug_ts = utils.aug_time_step_cont(time_step, self.skill)
        return self.policy.distribution(aug_ts, policy_state)