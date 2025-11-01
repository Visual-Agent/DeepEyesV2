# NOTE: Env must be imported here in order to trigger metaclass registering

from .envs.deepeyesv2.deepeyesv2 import DeepEyesV2_ENV

from .parallel_env import agent_rollout_loop
