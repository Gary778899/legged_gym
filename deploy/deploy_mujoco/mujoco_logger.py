from __future__ import annotations

import importlib
from typing import Iterable, Sequence

import numpy as np


class MujocoLogger:
    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        qpos_slice: slice = slice(7, 19),
        qvel_slice: slice = slice(6, 18),
        torque_slice: slice = slice(0, 12),
    ) -> None:
        self.joint_names = list(joint_names) if joint_names is not None else self._default_x2_joint_names()
        self.qpos_slice = qpos_slice
        self.qvel_slice = qvel_slice
        self.torque_slice = torque_slice
        self._validate_configuration()
        self.reset()

    @staticmethod
    def _default_x2_joint_names() -> list[str]:
        return [
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw",
            "left_knee",
            "left_ankle_pitch",
            "left_ankle_roll",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_knee",
            "right_ankle_pitch",
            "right_ankle_roll",
        ]

    @staticmethod
    def _slice_len(slice_obj: slice) -> int:
        if slice_obj.step not in (None, 1):
            raise ValueError("Only contiguous slices with step=1 are supported")
        if slice_obj.start is None or slice_obj.stop is None:
            raise ValueError("Slices must define explicit start and stop indices")
        if slice_obj.stop <= slice_obj.start:
            raise ValueError("Slice stop must be greater than slice start")
        return slice_obj.stop - slice_obj.start

    @staticmethod
    def _as_float32_copy(values: Iterable[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).copy()

    def _validate_configuration(self) -> None:
        qpos_len = self._slice_len(self.qpos_slice)
        qvel_len = self._slice_len(self.qvel_slice)
        torque_len = self._slice_len(self.torque_slice)
        names_len = len(self.joint_names)

        if qpos_len != qvel_len or qpos_len != torque_len:
            raise ValueError(
                "qpos_slice, qvel_slice, and torque_slice must select the same number of joints"
            )
        if names_len != qpos_len:
            raise ValueError(
                f"joint_names length ({names_len}) must match selected joint count ({qpos_len})"
            )

    def reset(self) -> None:
        self.times: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.actuator_force: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.times)

    def log_step(self, data) -> None:
        qpos = self._as_float32_copy(data.qpos[self.qpos_slice])
        qvel = self._as_float32_copy(data.qvel[self.qvel_slice])
        ctrl = self._as_float32_copy(data.ctrl[self.torque_slice])
        actuator_force = self._as_float32_copy(data.actuator_force[self.torque_slice])

        expected = len(self.joint_names)
        if not (len(qpos) == len(qvel) == len(ctrl) == len(actuator_force) == expected):
            raise ValueError(
                "Logged vector lengths do not match configured joint count; "
                f"expected {expected}, got qpos={len(qpos)}, qvel={len(qvel)}, "
                f"ctrl={len(ctrl)}, actuator_force={len(actuator_force)}"
            )

        self.times.append(float(data.time))
        self.qpos.append(qpos)
        self.qvel.append(qvel)
        self.ctrl.append(ctrl)
        self.actuator_force.append(actuator_force)

    def save_to_csv(self, filename: str):
        if len(self) == 0:
            raise ValueError("No samples were logged; call log_step(data) before saving")

        try:
            pd = importlib.import_module("pandas")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "pandas is required for CSV export. Install it with: pip install pandas"
            ) from exc

        rows = []
        for index in range(len(self)):
            row = {"time": self.times[index]}
            for joint_index, joint_name in enumerate(self.joint_names):
                row[f"qpos_{joint_name}"] = float(self.qpos[index][joint_index])
                row[f"qvel_{joint_name}"] = float(self.qvel[index][joint_index])
                row[f"ctrl_{joint_name}"] = float(self.ctrl[index][joint_index])
                row[f"actuator_force_{joint_name}"] = float(
                    self.actuator_force[index][joint_index]
                )
            rows.append(row)

        dataframe = pd.DataFrame(rows)
        dataframe.to_csv(filename, index=False)
        return dataframe
