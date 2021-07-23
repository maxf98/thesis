from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types

from core import utils


# wrapper class for policy to fix skill
class SkillConditionedPolicy(tf_policy.TFPolicy):
    def __init__(self, policy: tf_policy.TFPolicy, skill_prior: tfp.distributions.Distribution):
        super(SkillConditionedPolicy, self).__init__(
            policy.time_step_spec,
            policy.action_spec,
            policy.policy_state_spec,
            policy.info_spec
        )

        self.policy = policy
        self.skill_prior = skill_prior
        # this implies that for categorical distribution, sample must output one_hot codes
        self.skill = skill_prior.sample()

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        aug_ts = utils.aug_time_step(time_step, self.skill)
        return self.policy.distribution(aug_ts, policy_state)

    def resample_skill(self):
        self.skill = self.skill_prior.sample()


class FixedOptionPolicy:  # doesn't actually inherit from tfpolicy, its really just kind of a wrapper
    def __init__(self, policy, skill):
        self.policy = policy
        self.skill = skill

    def action(self, time_step: ts.TimeStep, policy_state: types.NestedTensor = (), seed: Optional[types.Seed] = None):
        aug_ts = utils.aug_time_step(time_step, self.skill)
        return self.policy.action(aug_ts)

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        aug_ts = utils.aug_time_step(time_step, self.skill)
        return self.policy.distribution(aug_ts, policy_state)