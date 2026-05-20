import numpy as np

from .constants import NUM_OBS
from .robot_state import RobotState, ScalingValues


def projected_gravity_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float64)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class ObservationBuilder:
    def __init__(self, default_dof_pos: np.ndarray, scales: ScalingValues):
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        self.scales = scales

    def build(self, state: RobotState, commands: np.ndarray, phase: float) -> np.ndarray:
        base_ang_vel = state.gyro * self.scales.ang_vel
        projected_gravity = projected_gravity_from_quaternion_xyzw(state.quat)
        dof_pos_obs = (state.joint_pos - self.default_dof_pos) * self.scales.dof_pos
        dof_vel_obs = state.joint_vel * self.scales.dof_vel

        obs = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                commands * self.scales.commands_scale,
                dof_pos_obs,
                dof_vel_obs,
                state.last_action,
                [np.sin(phase), np.cos(phase)],
            ]
        ).astype(np.float32).reshape(1, -1)

        if obs.shape[1] != NUM_OBS:
            raise ValueError(f"Expected observation width {NUM_OBS}, got {obs.shape[1]}")

        return obs
