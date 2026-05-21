from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import numpy as np
from aimdk_msgs.msg import JointCommandArray


@dataclass(frozen=True)
class CachedJointCommand:
    joint_order: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    sequence: int
    has_received_command: bool


class MujocoCommandAdapter:
    def __init__(
        self,
        node,
        topic_name: str,
        joint_order: Sequence[str],
        default_position: Sequence[float] | None = None,
        default_velocity: Sequence[float] | None = None,
        default_effort: Sequence[float] | None = None,
        default_stiffness: Sequence[float] | None = None,
        default_damping: Sequence[float] | None = None,
    ) -> None:
        self._node = node
        self._joint_order = tuple(joint_order)
        self._joint_name_to_index = {name: index for index, name in enumerate(self._joint_order)}
        self._num_joints = len(self._joint_order)
        self._lock = Lock()

        self._position = self._build_array(default_position)
        self._velocity = self._build_array(default_velocity)
        self._effort = self._build_array(default_effort)
        self._stiffness = self._build_array(default_stiffness)
        self._damping = self._build_array(default_damping)
        self._sequence = 0
        self._has_received_command = False

        self._subscription = node.create_subscription(
            JointCommandArray,
            topic_name,
            self._command_callback,
            10,
        )

    def get_latest_command(self) -> CachedJointCommand:
        with self._lock:
            return CachedJointCommand(
                joint_order=self._joint_order,
                position=self._position.copy(),
                velocity=self._velocity.copy(),
                effort=self._effort.copy(),
                stiffness=self._stiffness.copy(),
                damping=self._damping.copy(),
                sequence=self._sequence,
                has_received_command=self._has_received_command,
            )

    def _command_callback(self, msg: JointCommandArray) -> None:
        updated_joint_count = 0
        with self._lock:
            for joint_command in msg.joints:
                joint_index = self._joint_name_to_index.get(joint_command.name)
                if joint_index is None:
                    continue
                self._position[joint_index] = float(joint_command.position)
                self._velocity[joint_index] = float(joint_command.velocity)
                self._effort[joint_index] = float(joint_command.effort)
                self._stiffness[joint_index] = float(joint_command.stiffness)
                self._damping[joint_index] = float(joint_command.damping)
                updated_joint_count += 1
            self._sequence = int(msg.header.sequence)
            self._has_received_command = True

        if updated_joint_count != self._num_joints:
            self._node.get_logger().warn(
                "Received JointCommandArray with "
                f"{updated_joint_count}/{self._num_joints} known joints; keeping previous values for missing joints."
            )

    def _build_array(self, values: Sequence[float] | None) -> np.ndarray:
        if values is None:
            return np.zeros(self._num_joints, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._num_joints,):
            raise ValueError(
                f"Expected default array shape {(self._num_joints,)}, got {array.shape}"
            )
        return array.copy()
