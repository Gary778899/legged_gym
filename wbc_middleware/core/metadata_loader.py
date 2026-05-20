from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc_middleware.core.constants import (
    DEFAULT_METADATA_PATH,
    DEFAULT_POLICY_PATH,
    JOINT_ORDER,
    NUM_ACTIONS,
    NUM_OBS,
)
from wbc_middleware.core.robot_state import ScalingValues

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimeMetadata:
    num_observations: int
    num_actions: int
    rnn_hidden_size: int
    rnn_num_layers: int
    action_scale: float
    kp: np.ndarray
    kd: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    commands_scale: np.ndarray

    def build_scaling_values(self) -> ScalingValues:
        return ScalingValues(
            ang_vel=float(self.ang_vel_scale),
            dof_pos=float(self.dof_pos_scale),
            dof_vel=float(self.dof_vel_scale),
            commands_scale=np.asarray(self.commands_scale, dtype=np.float64),
        )


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    resolved_path = _resolve_project_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config mapping in {resolved_path}")
    return config


def load_runtime_metadata(
    *,
    policy_path: str | None = None,
    metadata_path: str | None = None,
    yaml_config: dict[str, Any] | None = None,
) -> RuntimeMetadata:
    yaml_config = yaml_config or {}
    raw_metadata = _load_onnx_embedded_metadata(policy_path or yaml_config.get("policy_path"))
    if raw_metadata is None:
        sidecar_path = metadata_path or yaml_config.get("metadata_path") or DEFAULT_METADATA_PATH
        raw_metadata = _load_metadata_sidecar(sidecar_path)

    metadata_payload = _extract_metadata_payload(raw_metadata)
    control_payload = metadata_payload.get("control", {}) if isinstance(metadata_payload, dict) else {}
    normalization_payload = metadata_payload.get("normalization", {}) if isinstance(metadata_payload, dict) else {}
    obs_scales_payload = (
        normalization_payload.get("obs_scales", {})
        if isinstance(normalization_payload, dict)
        else {}
    )

    num_observations = int(
        _coalesce(
            metadata_payload.get("num_observations") if isinstance(metadata_payload, dict) else None,
            yaml_config.get("num_observations"),
            yaml_config.get("num_obs"),
            NUM_OBS,
        )
    )
    num_actions = int(
        _coalesce(
            metadata_payload.get("num_actions") if isinstance(metadata_payload, dict) else None,
            yaml_config.get("num_actions"),
            NUM_ACTIONS,
        )
    )
    policy_payload = metadata_payload.get("policy", {}) if isinstance(metadata_payload, dict) else {}
    rnn_hidden_size = int(
        _coalesce(
            policy_payload.get("rnn_hidden_size") if isinstance(policy_payload, dict) else None,
            yaml_config.get("rnn_hidden_size"),
            64,
        )
    )
    rnn_num_layers = int(
        _coalesce(
            policy_payload.get("rnn_num_layers") if isinstance(policy_payload, dict) else None,
            yaml_config.get("rnn_num_layers"),
            1,
        )
    )

    action_scale = float(
        _coalesce(
            control_payload.get("action_scale") if isinstance(control_payload, dict) else None,
            yaml_config.get("action_scale"),
            0.25,
        )
    )
    kp = _resolve_joint_vector(
        control_payload.get("kp") if isinstance(control_payload, dict) else None,
        yaml_config.get("kps"),
        default_value=0.0,
    )
    kd = _resolve_joint_vector(
        control_payload.get("kd") if isinstance(control_payload, dict) else None,
        yaml_config.get("kds"),
        default_value=0.0,
    )
    ang_vel_scale = float(
        _coalesce(
            obs_scales_payload.get("ang_vel") if isinstance(obs_scales_payload, dict) else None,
            yaml_config.get("ang_vel_scale"),
            0.25,
        )
    )
    dof_pos_scale = float(
        _coalesce(
            obs_scales_payload.get("dof_pos") if isinstance(obs_scales_payload, dict) else None,
            yaml_config.get("dof_pos_scale"),
            1.0,
        )
    )
    dof_vel_scale = float(
        _coalesce(
            obs_scales_payload.get("dof_vel") if isinstance(obs_scales_payload, dict) else None,
            yaml_config.get("dof_vel_scale"),
            0.05,
        )
    )
    commands_scale = np.asarray(
        _coalesce(
            obs_scales_payload.get("commands") if isinstance(obs_scales_payload, dict) else None,
            yaml_config.get("commands_scale"),
            yaml_config.get("cmd_scale"),
            [2.0, 2.0, 0.25],
        ),
        dtype=np.float64,
    )
    if commands_scale.shape != (3,):
        raise ValueError(f"Expected commands_scale with shape (3,), got {commands_scale.shape}")

    return RuntimeMetadata(
        num_observations=num_observations,
        num_actions=num_actions,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        action_scale=action_scale,
        kp=kp,
        kd=kd,
        ang_vel_scale=ang_vel_scale,
        dof_pos_scale=dof_pos_scale,
        dof_vel_scale=dof_vel_scale,
        commands_scale=commands_scale,
    )


def _load_onnx_embedded_metadata(policy_path: str | None) -> dict[str, Any] | None:
    if not policy_path:
        policy_path = DEFAULT_POLICY_PATH
    try:
        import onnx
    except Exception:
        return None

    onnx_path = _resolve_project_path(policy_path)
    if not onnx_path.exists():
        return None

    model = onnx.load(str(onnx_path))
    if not model.metadata_props:
        return None

    payload: dict[str, Any] = {}
    for entry in model.metadata_props:
        try:
            payload[entry.key] = json.loads(entry.value)
        except Exception:
            payload[entry.key] = entry.value
    return payload


def _load_metadata_sidecar(metadata_path: str | Path) -> dict[str, Any] | None:
    candidate = _resolve_project_path(metadata_path)
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _extract_metadata_payload(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
        return metadata["metadata"]
    return metadata


def _resolve_joint_vector(
    metadata_value: Any,
    yaml_value: Any,
    *,
    default_value: float,
) -> np.ndarray:
    if metadata_value is not None:
        if isinstance(metadata_value, dict):
            values = [_resolve_grouped_joint_value(name, metadata_value) for name in JOINT_ORDER]
            return np.asarray(values, dtype=np.float64)
        metadata_arr = np.asarray(metadata_value, dtype=np.float64)
        if metadata_arr.shape == (len(JOINT_ORDER),):
            return metadata_arr

    if yaml_value is not None:
        yaml_arr = np.asarray(yaml_value, dtype=np.float64)
        if yaml_arr.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Expected per-joint vector of length {len(JOINT_ORDER)}, got shape {yaml_arr.shape}"
            )
        return yaml_arr

    return np.full(len(JOINT_ORDER), float(default_value), dtype=np.float64)


def _resolve_grouped_joint_value(joint_name: str, grouped_values: dict[str, Any]) -> float:
    canonical_name = joint_name
    if canonical_name.startswith("left_"):
        canonical_name = canonical_name[len("left_") :]
    elif canonical_name.startswith("right_"):
        canonical_name = canonical_name[len("right_") :]
    if canonical_name not in grouped_values:
        raise KeyError(f"Missing grouped joint parameter for '{canonical_name}'")
    return float(grouped_values[canonical_name])


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_project_path(path_value: str | Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()
