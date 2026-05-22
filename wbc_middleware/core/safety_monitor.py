from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import asin, atan2
from typing import Any

import numpy as np

from wbc_middleware.core.command_mapper import SafetyLimits
from wbc_middleware.core.robot_state import RobotState


class SafetySeverity(str, Enum):
    STOPPING = "STOPPING"
    SAFE_HOLD = "SAFE_HOLD"


@dataclass(frozen=True)
class ThresholdPair:
    stopping: float
    safe_hold: float


@dataclass(frozen=True)
class SafetyDecision:
    severity: SafetySeverity
    reason: str


@dataclass(frozen=True)
class SafetyMonitorConfig:
    enabled: bool
    check_states: frozenset[str]
    nominal_pose_check_states: frozenset[str]
    tracking_error_check_states: frozenset[str]
    quaternion_norm_tolerance: float
    roll_limit: ThresholdPair
    pitch_limit: ThresholdPair
    angular_velocity_limit: tuple[np.ndarray, np.ndarray]
    nominal_pose_error_limit: ThresholdPair
    tracking_error_limit: ThresholdPair | None
    joint_limit_violation_limit: ThresholdPair | None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        num_joints: int,
    ) -> "SafetyMonitorConfig":
        config = config or {}
        enabled = bool(config.get("enabled", True))
        check_states = _resolve_states(
            config.get("check_states"),
            default=("MOVE_TO_DEFAULT", "POLICY_HOLD", "POLICY_ACTIVE"),
        )
        nominal_pose_check_states = _resolve_states(
            config.get("nominal_pose_check_states"),
            default=("POLICY_HOLD",),
        )
        tracking_error_check_states = _resolve_states(
            config.get("tracking_error_check_states"),
            default=("MOVE_TO_DEFAULT", "POLICY_HOLD"),
        )
        imu_config = _resolve_mapping(config.get("imu"), field_name="safety.monitor.imu")
        joints_config = _resolve_mapping(
            config.get("joints"),
            field_name="safety.monitor.joints",
        )
        return cls(
            enabled=enabled,
            check_states=check_states,
            nominal_pose_check_states=nominal_pose_check_states,
            tracking_error_check_states=tracking_error_check_states,
            quaternion_norm_tolerance=float(
                abs(imu_config.get("quaternion_norm_tolerance", 0.15))
            ),
            roll_limit=_resolve_threshold_pair(
                imu_config.get("roll"),
                field_name="safety.monitor.imu.roll",
                default_stopping=0.45,
                default_safe_hold=0.7,
            ),
            pitch_limit=_resolve_threshold_pair(
                imu_config.get("pitch"),
                field_name="safety.monitor.imu.pitch",
                default_stopping=0.45,
                default_safe_hold=0.7,
            ),
            angular_velocity_limit=(
                _resolve_vector_threshold(
                    imu_config.get("angular_velocity"),
                    key="stopping",
                    field_name="safety.monitor.imu.angular_velocity.stopping",
                    num_joints=3,
                    default=(1.8, 1.8, 2.5),
                ),
                _resolve_vector_threshold(
                    imu_config.get("angular_velocity"),
                    key="safe_hold",
                    field_name="safety.monitor.imu.angular_velocity.safe_hold",
                    num_joints=3,
                    default=(3.2, 3.2, 4.5),
                ),
            ),
            nominal_pose_error_limit=_resolve_threshold_pair(
                joints_config.get("nominal_pose_error"),
                field_name="safety.monitor.joints.nominal_pose_error",
                default_stopping=0.45,
                default_safe_hold=0.8,
            ),
            tracking_error_limit=_resolve_optional_threshold_pair(
                joints_config.get("tracking_error"),
                field_name="safety.monitor.joints.tracking_error",
                default_stopping=0.35,
                default_safe_hold=0.6,
            ),
            joint_limit_violation_limit=_resolve_optional_threshold_pair(
                joints_config.get("position_limit_violation"),
                field_name="safety.monitor.joints.position_limit_violation",
                default_stopping=0.03,
                default_safe_hold=0.08,
            ),
        )


class SafetyMonitor:
    def __init__(
        self,
        config: SafetyMonitorConfig,
        *,
        default_dof_pos: np.ndarray,
        safety_limits: SafetyLimits,
    ):
        self._config = config
        self._default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        self._joint_position_lower = _copy_optional_array(safety_limits.joint_position_lower)
        self._joint_position_upper = _copy_optional_array(safety_limits.joint_position_upper)

    def evaluate(
        self,
        *,
        state: RobotState,
        startup_state: str,
        commanded_joint_positions: np.ndarray | None,
    ) -> SafetyDecision | None:
        if not self._config.enabled or startup_state not in self._config.check_states:
            return None

        sensor_validity_decision = self._check_sensor_validity(state)
        if sensor_validity_decision is not None:
            return sensor_validity_decision

        roll, pitch = self._torso_roll_pitch(state.quat)

        decision = self._check_scalar_abs(
            value=roll,
            thresholds=self._config.roll_limit,
            label="torso roll",
        )
        if decision is not None:
            return decision

        decision = self._check_scalar_abs(
            value=pitch,
            thresholds=self._config.pitch_limit,
            label="torso pitch",
        )
        if decision is not None:
            return decision

        decision = self._check_vector_abs(
            vector=state.gyro,
            stopping_limit=self._config.angular_velocity_limit[0],
            safe_hold_limit=self._config.angular_velocity_limit[1],
            labels=("gyro x", "gyro y", "gyro z"),
        )
        if decision is not None:
            return decision

        decision = self._check_joint_limit_violation(state.joint_pos)
        if decision is not None:
            return decision

        if startup_state in self._config.nominal_pose_check_states:
            decision = self._check_joint_error(
                actual=state.joint_pos,
                reference=self._default_dof_pos,
                thresholds=self._config.nominal_pose_error_limit,
                label="joint nominal-pose deviation",
            )
            if decision is not None:
                return decision

        if (
            commanded_joint_positions is not None
            and self._config.tracking_error_limit is not None
            and startup_state in self._config.tracking_error_check_states
        ):
            decision = self._check_joint_error(
                actual=state.joint_pos,
                reference=commanded_joint_positions,
                thresholds=self._config.tracking_error_limit,
                label="joint tracking error",
            )
            if decision is not None:
                return decision

        return None

    def _check_sensor_validity(self, state: RobotState) -> SafetyDecision | None:
        if not np.all(np.isfinite(state.quat)):
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason="IMU quaternion contains NaN/Inf",
            )
        if not np.all(np.isfinite(state.gyro)):
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason="IMU angular velocity contains NaN/Inf",
            )
        if not np.all(np.isfinite(state.joint_pos)):
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason="joint positions contain NaN/Inf",
            )
        if not np.all(np.isfinite(state.joint_vel)):
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason="joint velocities contain NaN/Inf",
            )

        quat_norm = float(np.linalg.norm(state.quat))
        if abs(quat_norm - 1.0) > self._config.quaternion_norm_tolerance:
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason=(
                    "IMU quaternion norm out of range: "
                    f"norm={quat_norm:.3f}, tolerance={self._config.quaternion_norm_tolerance:.3f}"
                ),
            )
        return None

    def _check_joint_limit_violation(self, joint_pos: np.ndarray) -> SafetyDecision | None:
        thresholds = self._config.joint_limit_violation_limit
        if thresholds is None:
            return None

        violation = np.zeros_like(joint_pos, dtype=np.float64)
        if self._joint_position_lower is not None:
            violation = np.maximum(violation, self._joint_position_lower - joint_pos)
        if self._joint_position_upper is not None:
            violation = np.maximum(violation, joint_pos - self._joint_position_upper)

        max_violation = float(np.max(violation)) if violation.size else 0.0
        if max_violation <= thresholds.stopping:
            return None

        severity = (
            SafetySeverity.SAFE_HOLD
            if max_violation > thresholds.safe_hold
            else SafetySeverity.STOPPING
        )
        return SafetyDecision(
            severity=severity,
            reason=(
                "joint position limit violation exceeded threshold: "
                f"max_violation={max_violation:.3f} rad"
            ),
        )

    def _check_joint_error(
        self,
        *,
        actual: np.ndarray,
        reference: np.ndarray,
        thresholds: ThresholdPair,
        label: str,
    ) -> SafetyDecision | None:
        error = np.abs(actual - reference)
        max_error = float(np.max(error)) if error.size else 0.0
        if max_error <= thresholds.stopping:
            return None
        severity = (
            SafetySeverity.SAFE_HOLD
            if max_error > thresholds.safe_hold
            else SafetySeverity.STOPPING
        )
        return SafetyDecision(
            severity=severity,
            reason=f"{label} exceeded threshold: max_error={max_error:.3f} rad",
        )

    def _check_scalar_abs(
        self,
        *,
        value: float,
        thresholds: ThresholdPair,
        label: str,
    ) -> SafetyDecision | None:
        magnitude = abs(float(value))
        if magnitude <= thresholds.stopping:
            return None
        severity = (
            SafetySeverity.SAFE_HOLD
            if magnitude > thresholds.safe_hold
            else SafetySeverity.STOPPING
        )
        return SafetyDecision(
            severity=severity,
            reason=f"{label} exceeded threshold: abs_value={magnitude:.3f}",
        )

    def _check_vector_abs(
        self,
        *,
        vector: np.ndarray,
        stopping_limit: np.ndarray,
        safe_hold_limit: np.ndarray,
        labels: tuple[str, ...],
    ) -> SafetyDecision | None:
        magnitudes = np.abs(np.asarray(vector, dtype=np.float64))
        safe_hold_mask = magnitudes > safe_hold_limit
        if np.any(safe_hold_mask):
            index = int(np.argmax(magnitudes - safe_hold_limit))
            return SafetyDecision(
                severity=SafetySeverity.SAFE_HOLD,
                reason=(
                    f"{labels[index]} exceeded threshold: "
                    f"abs_value={magnitudes[index]:.3f}"
                ),
            )
        stopping_mask = magnitudes > stopping_limit
        if np.any(stopping_mask):
            index = int(np.argmax(magnitudes - stopping_limit))
            return SafetyDecision(
                severity=SafetySeverity.STOPPING,
                reason=(
                    f"{labels[index]} exceeded threshold: "
                    f"abs_value={magnitudes[index]:.3f}"
                ),
            )
        return None

    @staticmethod
    def _torso_roll_pitch(quat_xyzw: np.ndarray) -> tuple[float, float]:
        x, y, z, w = (float(value) for value in quat_xyzw)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = asin(sinp)
        return roll, pitch


def _resolve_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _resolve_states(value: Any, *, default: tuple[str, ...]) -> frozenset[str]:
    if value is None:
        return frozenset(default)
    if not isinstance(value, list):
        raise ValueError("safety.monitor state lists must be arrays")
    return frozenset(str(item) for item in value)


def _resolve_threshold_pair(
    value: Any,
    *,
    field_name: str,
    default_stopping: float,
    default_safe_hold: float,
) -> ThresholdPair:
    mapping = _resolve_mapping(value, field_name=field_name)
    stopping = float(abs(mapping.get("stopping", default_stopping)))
    safe_hold = float(abs(mapping.get("safe_hold", default_safe_hold)))
    if safe_hold < stopping:
        raise ValueError(f"{field_name}.safe_hold must be >= {field_name}.stopping")
    return ThresholdPair(stopping=stopping, safe_hold=safe_hold)


def _resolve_optional_threshold_pair(
    value: Any,
    *,
    field_name: str,
    default_stopping: float,
    default_safe_hold: float,
) -> ThresholdPair | None:
    if value is None:
        return None
    return _resolve_threshold_pair(
        value,
        field_name=field_name,
        default_stopping=default_stopping,
        default_safe_hold=default_safe_hold,
    )


def _resolve_vector_threshold(
    value: Any,
    *,
    key: str,
    field_name: str,
    num_joints: int,
    default: tuple[float, ...],
) -> np.ndarray:
    mapping = _resolve_mapping(value, field_name=field_name.rsplit(".", 1)[0])
    raw_vector = mapping.get(key, default)
    array = np.asarray(raw_vector, dtype=np.float64)
    if array.shape != (num_joints,):
        raise ValueError(f"{field_name} must have shape {(num_joints,)}, got {array.shape}")
    return np.abs(array)


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).copy()
