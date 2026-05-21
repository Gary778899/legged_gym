from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wbc_middleware.core.constants import JOINT_ORDER, UPPER_BODY_JOINT_ORDER


@dataclass(frozen=True)
class UpperBodyHoldProfile:
    kp: np.ndarray
    kd: np.ndarray


@dataclass(frozen=True)
class UpperBodyStabilizationConfig:
    enabled: bool
    waist_pitch_kp: float
    waist_pitch_kd: float
    waist_roll_kp: float
    waist_roll_kd: float
    max_pitch_offset: float
    max_roll_offset: float
    waist_kp_scale: float
    waist_kd_scale: float
    arm_kp_scale: float
    arm_kd_scale: float
    head_kp_scale: float
    head_kd_scale: float


@dataclass(frozen=True)
class UpperBodyConfig:
    enabled: bool
    publish_in_mock: bool
    joint_names: tuple[str, ...]
    default_angles: np.ndarray
    active_hold: UpperBodyHoldProfile
    safe_hold_mode: str
    safe_hold: UpperBodyHoldProfile
    stabilization: UpperBodyStabilizationConfig

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> 'UpperBodyConfig':
        config = config or {}
        section = config.get('upper_body', {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError('upper_body must be a mapping when provided')

        enabled = bool(section.get('enabled', False))
        publish_in_mock = bool(section.get('publish_in_mock', False))
        raw_joint_names = section.get('joint_names', [])
        if raw_joint_names is None:
            raw_joint_names = []
        if not isinstance(raw_joint_names, list):
            raise ValueError('upper_body.joint_names must be a list')

        stabilization = _resolve_stabilization_config(section.get('stabilization'))

        if not raw_joint_names:
            empty = np.zeros(0, dtype=np.float64)
            return cls(
                enabled=enabled,
                publish_in_mock=publish_in_mock,
                joint_names=(),
                default_angles=empty.copy(),
                active_hold=UpperBodyHoldProfile(kp=empty.copy(), kd=empty.copy()),
                safe_hold_mode='hold_position',
                safe_hold=UpperBodyHoldProfile(kp=empty.copy(), kd=empty.copy()),
                stabilization=stabilization,
            )

        duplicate_names = _find_duplicates(raw_joint_names)
        if duplicate_names:
            raise ValueError(f'upper_body.joint_names contains duplicates: {duplicate_names}')

        lower_body_overlap = [name for name in raw_joint_names if name in JOINT_ORDER]
        if lower_body_overlap:
            raise ValueError(
                'upper_body.joint_names must not overlap lower-body policy joints: '
                f'{lower_body_overlap}'
            )

        unknown_joints = [name for name in raw_joint_names if name not in UPPER_BODY_JOINT_ORDER]
        if unknown_joints:
            raise ValueError(
                'upper_body.joint_names contains joints unknown to the vendor upper-body model: '
                f'{unknown_joints}'
            )

        joint_name_set = set(raw_joint_names)
        joint_names = tuple(name for name in UPPER_BODY_JOINT_ORDER if name in joint_name_set)
        reorder_index = [raw_joint_names.index(name) for name in joint_names]
        num_joints = len(raw_joint_names)

        default_angles = _resolve_vector(
            section.get('default_angles'),
            field_name='upper_body.default_angles',
            expected_size=num_joints,
        )[reorder_index]
        active_hold = _resolve_hold_profile(
            section.get('active_hold'),
            field_prefix='upper_body.active_hold',
            expected_size=num_joints,
            reorder_index=reorder_index,
        )
        safe_hold_section = section.get('safe_hold', {}) or {}
        if not isinstance(safe_hold_section, dict):
            raise ValueError('upper_body.safe_hold must be a mapping')
        safe_hold_mode = str(safe_hold_section.get('mode', 'hold_position')).lower()
        if safe_hold_mode != 'hold_position':
            raise ValueError('upper_body.safe_hold.mode currently only supports hold_position')
        safe_hold = _resolve_hold_profile(
            safe_hold_section,
            field_prefix='upper_body.safe_hold',
            expected_size=num_joints,
            reorder_index=reorder_index,
        )

        return cls(
            enabled=enabled,
            publish_in_mock=publish_in_mock,
            joint_names=joint_names,
            default_angles=default_angles,
            active_hold=active_hold,
            safe_hold_mode=safe_hold_mode,
            safe_hold=safe_hold,
            stabilization=stabilization,
        )

    def should_publish(self, *, is_mock_runtime: bool) -> bool:
        if not self.enabled or not self.joint_names:
            return False
        if is_mock_runtime and not self.publish_in_mock:
            return False
        return True


def _resolve_hold_profile(
    payload: dict[str, Any] | None,
    *,
    field_prefix: str,
    expected_size: int,
    reorder_index: list[int],
) -> UpperBodyHoldProfile:
    if not isinstance(payload, dict):
        raise ValueError(f'{field_prefix} must be a mapping')
    kp = _resolve_vector(
        payload.get('kp'),
        field_name=f'{field_prefix}.kp',
        expected_size=expected_size,
    )[reorder_index]
    kd = _resolve_vector(
        payload.get('kd'),
        field_name=f'{field_prefix}.kd',
        expected_size=expected_size,
    )[reorder_index]
    return UpperBodyHoldProfile(kp=kp, kd=kd)


def _resolve_stabilization_config(payload: dict[str, Any] | None) -> UpperBodyStabilizationConfig:
    payload = payload or {}
    if not isinstance(payload, dict):
        raise ValueError('upper_body.stabilization must be a mapping when provided')
    return UpperBodyStabilizationConfig(
        enabled=bool(payload.get('enabled', False)),
        waist_pitch_kp=float(payload.get('waist_pitch_kp', 0.0)),
        waist_pitch_kd=float(payload.get('waist_pitch_kd', 0.0)),
        waist_roll_kp=float(payload.get('waist_roll_kp', 0.0)),
        waist_roll_kd=float(payload.get('waist_roll_kd', 0.0)),
        max_pitch_offset=float(abs(payload.get('max_pitch_offset', 0.0))),
        max_roll_offset=float(abs(payload.get('max_roll_offset', 0.0))),
        waist_kp_scale=float(payload.get('waist_kp_scale', 1.0)),
        waist_kd_scale=float(payload.get('waist_kd_scale', 1.0)),
        arm_kp_scale=float(payload.get('arm_kp_scale', 1.0)),
        arm_kd_scale=float(payload.get('arm_kd_scale', 1.0)),
        head_kp_scale=float(payload.get('head_kp_scale', 1.0)),
        head_kd_scale=float(payload.get('head_kd_scale', 1.0)),
    )


def _resolve_vector(value: Any, *, field_name: str, expected_size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_size,):
        raise ValueError(f'{field_name} must have shape {(expected_size,)}, got {array.shape}')
    return array.copy()


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
