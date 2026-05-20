from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wbc_middleware.core.constants import JOINT_ORDER
from wbc_middleware.core.metadata_loader import RuntimeMetadata


@dataclass(frozen=True)
class JointCommandTarget:
    name: str
    position: float
    velocity: float
    effort: float
    stiffness: float
    damping: float


class CommandMapper:
    def __init__(self, default_dof_pos: np.ndarray, metadata: RuntimeMetadata):
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        self.metadata = metadata

    def map_action_to_targets(self, action: np.ndarray) -> list[JointCommandTarget]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(JOINT_ORDER),):
            raise ValueError(f"Expected action shape {(len(JOINT_ORDER),)}, got {action.shape}")
        target_positions = self.default_dof_pos + action * self.metadata.action_scale
        return self.build_position_targets(target_positions)

    def build_default_pose_targets(self) -> list[JointCommandTarget]:
        return self.build_position_targets(self.default_dof_pos)

    def build_position_targets(self, target_positions: np.ndarray) -> list[JointCommandTarget]:
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Expected target_positions shape {(len(JOINT_ORDER),)}, got {target_positions.shape}"
            )
        return [
            JointCommandTarget(
                name=joint_name,
                position=float(target_positions[index]),
                velocity=0.0,
                effort=0.0,
                stiffness=float(self.metadata.kp[index]),
                damping=float(self.metadata.kd[index]),
            )
            for index, joint_name in enumerate(JOINT_ORDER)
        ]
