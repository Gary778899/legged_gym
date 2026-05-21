from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.x2.x2_config import X2RoughCfg, X2RoughCfgPPO
from legged_gym.envs.x2.x2_env import X2Robot
from legged_gym.envs.x2.x2_fullbody_config import X2FullbodyCfg, X2FullbodyCfgPPO
from legged_gym.envs.x2.x2_fullbody_env import X2FullBodyRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "x2", X2Robot, X2RoughCfg(), X2RoughCfgPPO())
task_registry.register( "x2_fullbody", X2FullBodyRobot, X2FullbodyCfg(), X2FullbodyCfgPPO())
