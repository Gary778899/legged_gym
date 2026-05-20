from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class SafetyLimits:
    action_clip: float
    position_delta_clip: float
    joint_position_lower: np.ndarray | None
    joint_position_upper: np.ndarray | None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        num_joints: int,
        default_action_clip: float,
        default_position_delta_clip: float,
    ) -> "SafetyLimits":
        config = config or {}
        joint_position_lower = _resolve_optional_joint_limits(
            config.get("joint_position_lower"),
            num_joints=num_joints,
        )
        joint_position_upper = _resolve_optional_joint_limits(
            config.get("joint_position_upper"),
            num_joints=num_joints,
        )
        return cls(
            action_clip=float(abs(config.get("action_clip", default_action_clip))),
            position_delta_clip=float(
                abs(config.get("position_delta_clip", default_position_delta_clip))
            ),
            joint_position_lower=joint_position_lower,
            joint_position_upper=joint_position_upper,
        )


class CommandMapper:
    def __init__(
        self,
        default_dof_pos: np.ndarray,
        metadata: RuntimeMetadata,
        *,
        safety_limits: SafetyLimits | None = None,
    ):
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        self.metadata = metadata
        self.safety_limits = safety_limits or SafetyLimits(
            action_clip=float(abs(metadata.clip_actions)),
            position_delta_clip=float(abs(metadata.action_scale * metadata.clip_actions)),
            joint_position_lower=None,
            joint_position_upper=None,
        )

    def map_action_to_targets(self, action: np.ndarray) -> list[JointCommandTarget]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(JOINT_ORDER),):
            raise ValueError(f"Expected action shape {(len(JOINT_ORDER),)}, got {action.shape}")
        safe_action = self._clip_action(action)
        target_delta = np.clip(
            safe_action * self.metadata.action_scale,
            -self.safety_limits.position_delta_clip,
            self.safety_limits.position_delta_clip,
        )
        target_positions = self.default_dof_pos + target_delta
        return self.build_position_targets(target_positions)

    def build_default_pose_targets(self) -> list[JointCommandTarget]:
        return self.build_position_targets(self.default_dof_pos)

    def build_damping_targets(
        self,
        *,
        damping: np.ndarray | None = None,
        position_reference: np.ndarray | None = None,
    ) -> list[JointCommandTarget]:
        damping_values = self.metadata.kd if damping is None else np.asarray(damping, dtype=np.float64)
        if damping_values.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Expected damping vector shape {(len(JOINT_ORDER),)}, got {damping_values.shape}"
            )
        position_values = (
            self.default_dof_pos
            if position_reference is None
            else np.asarray(position_reference, dtype=np.float64)
        )
        if position_values.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Expected position_reference shape {(len(JOINT_ORDER),)}, got {position_values.shape}"
            )
        return [
            JointCommandTarget(
                name=joint_name,
                position=float(position_values[index]),
                velocity=0.0,
                effort=0.0,
                stiffness=0.0,
                damping=float(damping_values[index]),
            )
            for index, joint_name in enumerate(JOINT_ORDER)
        ]

    def build_position_targets(self, target_positions: np.ndarray) -> list[JointCommandTarget]:
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Expected target_positions shape {(len(JOINT_ORDER),)}, got {target_positions.shape}"
            )
        target_positions = self._clip_joint_positions(target_positions)
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

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        action_clip = self.safety_limits.action_clip
        if action_clip <= 0.0:
            return np.zeros_like(action)
        return np.clip(action, -action_clip, action_clip)

    def _clip_joint_positions(self, target_positions: np.ndarray) -> np.ndarray:
        if self.safety_limits.joint_position_lower is not None:
            target_positions = np.maximum(target_positions, self.safety_limits.joint_position_lower)
        if self.safety_limits.joint_position_upper is not None:
            target_positions = np.minimum(target_positions, self.safety_limits.joint_position_upper)
        return target_positions


def _resolve_optional_joint_limits(
    value: Any,
    *,
    num_joints: int,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (num_joints,):
        raise ValueError(f"Expected joint limit vector shape {(num_joints,)}, got {array.shape}")
    return array
