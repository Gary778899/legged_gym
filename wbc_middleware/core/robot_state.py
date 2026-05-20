from dataclasses import dataclass, field

import numpy as np

from .constants import JOINT_ORDER, NUM_ACTIONS


DEFAULT_JOINT_DICT = {
    "left_hip_pitch_joint": -0.248,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.5303,
    "left_ankle_pitch_joint": -0.2823,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.248,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.5303,
    "right_ankle_pitch_joint": -0.2823,
    "right_ankle_roll_joint": 0.0,
}


@dataclass
class ScalingValues:
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    commands_scale: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 0.25], dtype=np.float64)
    )


@dataclass
class RobotState:
    quat: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    joint_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    joint_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    last_action: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    default_joint_dict: dict[str, float] = field(
        default_factory=lambda: DEFAULT_JOINT_DICT.copy()
    )

    def default_dof_pos(self) -> np.ndarray:
        return np.array(
            [self.default_joint_dict[name] for name in JOINT_ORDER], dtype=np.float64
        )
