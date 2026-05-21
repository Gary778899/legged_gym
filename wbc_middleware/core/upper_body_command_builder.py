from __future__ import annotations

from typing import Sequence

import numpy as np

from wbc_middleware.core.command_mapper import JointCommandTarget
from wbc_middleware.core.constants import ARM_JOINT_ORDER, HEAD_JOINT_ORDER, WAIST_JOINT_ORDER
from wbc_middleware.core.upper_body_config import UpperBodyConfig, UpperBodyHoldProfile


class UpperBodyCommandBuilder:
    def __init__(self, upper_body_config: UpperBodyConfig):
        self._config = upper_body_config
        self._joint_name_to_index = {
            joint_name: index for index, joint_name in enumerate(self._config.joint_names)
        }

    def build_active_hold_targets(
        self,
        *,
        torso_roll: float = 0.0,
        torso_pitch: float = 0.0,
        torso_roll_rate: float = 0.0,
        torso_pitch_rate: float = 0.0,
    ) -> list[JointCommandTarget]:
        return self._build_targets(
            self._config.active_hold,
            active_stabilization={
                'torso_roll': torso_roll,
                'torso_pitch': torso_pitch,
                'torso_roll_rate': torso_roll_rate,
                'torso_pitch_rate': torso_pitch_rate,
            },
        )

    def build_safe_hold_targets(self) -> list[JointCommandTarget]:
        return self._build_targets(self._config.safe_hold)

    def _build_targets(
        self,
        profile: UpperBodyHoldProfile,
        *,
        active_stabilization: dict[str, float] | None = None,
    ) -> list[JointCommandTarget]:
        active_stabilization = active_stabilization or {}
        stabilization = self._config.stabilization
        targets: list[JointCommandTarget] = []
        for index, joint_name in enumerate(self._config.joint_names):
            position = float(self._config.default_angles[index])
            stiffness = float(profile.kp[index])
            damping = float(profile.kd[index])

            if profile is self._config.active_hold:
                if joint_name in WAIST_JOINT_ORDER:
                    stiffness *= stabilization.waist_kp_scale
                    damping *= stabilization.waist_kd_scale
                elif joint_name in ARM_JOINT_ORDER:
                    stiffness *= stabilization.arm_kp_scale
                    damping *= stabilization.arm_kd_scale
                elif joint_name in HEAD_JOINT_ORDER:
                    stiffness *= stabilization.head_kp_scale
                    damping *= stabilization.head_kd_scale

                if stabilization.enabled:
                    if joint_name == 'waist_pitch_joint':
                        position += self._clip_offset(
                            -(stabilization.waist_pitch_kp * active_stabilization.get('torso_pitch', 0.0)
                              + stabilization.waist_pitch_kd * active_stabilization.get('torso_pitch_rate', 0.0)),
                            stabilization.max_pitch_offset,
                        )
                    elif joint_name == 'waist_roll_joint':
                        position += self._clip_offset(
                            -(stabilization.waist_roll_kp * active_stabilization.get('torso_roll', 0.0)
                              + stabilization.waist_roll_kd * active_stabilization.get('torso_roll_rate', 0.0)),
                            stabilization.max_roll_offset,
                        )

            targets.append(
                JointCommandTarget(
                    name=joint_name,
                    position=position,
                    velocity=0.0,
                    effort=0.0,
                    stiffness=stiffness,
                    damping=damping,
                )
            )
        return targets

    def _clip_offset(self, value: float, limit: float) -> float:
        if limit <= 0.0:
            return 0.0
        return float(np.clip(value, -limit, limit))


def split_upper_body_targets_by_area(
    targets: Sequence[JointCommandTarget],
) -> dict[str, list[JointCommandTarget]]:
    area_orders = {
        'waist': tuple(WAIST_JOINT_ORDER),
        'arm': tuple(ARM_JOINT_ORDER),
        'head': tuple(HEAD_JOINT_ORDER),
    }
    joint_to_area = {
        joint_name: area_name
        for area_name, joint_order in area_orders.items()
        for joint_name in joint_order
    }
    grouped_lookup = {area_name: {} for area_name in area_orders}

    for target in targets:
        area_name = joint_to_area.get(target.name)
        if area_name is None:
            raise ValueError(
                f'Joint target {target.name!r} is outside the vendor SDK upper-body model'
            )
        grouped_lookup[area_name][target.name] = target

    grouped_targets: dict[str, list[JointCommandTarget]] = {}
    for area_name, joint_order in area_orders.items():
        ordered_targets = [
            grouped_lookup[area_name][joint_name]
            for joint_name in joint_order
            if joint_name in grouped_lookup[area_name]
        ]
        if ordered_targets:
            grouped_targets[area_name] = ordered_targets
    return grouped_targets
