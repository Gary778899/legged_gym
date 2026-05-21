from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from aimdk_msgs.msg import DomainErrorState, JointState, JointStateArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from wbc_middleware.core.constants import (
    ARM_JOINT_ORDER,
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HEAD_COMMAND_TOPIC,
    DEFAULT_IMU_TOPIC,
    DEFAULT_JOINT_COMMAND_TOPIC,
    DEFAULT_JOINT_STATE_TOPIC,
    DEFAULT_WAIST_COMMAND_TOPIC,
    HEAD_JOINT_ORDER,
    JOINT_ORDER,
    WAIST_JOINT_ORDER,
)
from wbc_middleware.core.metadata_loader import load_yaml_config
from wbc_middleware.core.upper_body_config import UpperBodyConfig
from wbc_middleware.mock_bridge.mujoco_command_adapter import MujocoCommandAdapter


@dataclass(frozen=True)
class JointGroupSpec:
    name: str
    topic_name: str
    joint_order: tuple[str, ...]
    default_position: np.ndarray
    default_stiffness: np.ndarray
    default_damping: np.ndarray


@dataclass(frozen=True)
class JointGroupRuntime:
    spec: JointGroupSpec
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    actuator_indices: np.ndarray
    actuator_ctrl_range: np.ndarray
    adapter: MujocoCommandAdapter


class MujocoHalBridge(Node):
    def __init__(self, config_path: str | None = None, *, enable_viewer: bool = False) -> None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = load_yaml_config(config_path)
        bridge_config = dict(self._config.get("mock_bridge", {}))
        node_name = str(bridge_config.get("node_name", "mujoco_hal_bridge"))
        super().__init__(node_name)

        self._project_root = Path(__file__).resolve().parents[2]
        self._leg_joint_order = tuple(JOINT_ORDER)
        self._num_joints = len(self._leg_joint_order)

        self._simulation_dt = float(bridge_config["simulation_dt"])
        self._state_publish_period_s = float(bridge_config["state_publish_period_s"])
        self._control_period_s = float(bridge_config["control_period_s"])
        self._hold_nominal_pose_until_first_command = bool(
            bridge_config.get("hold_nominal_pose_until_first_command", False)
        )
        self._default_angles = self._as_vector(
            self._config.get("default_angles"),
            "default_angles",
            expected_size=self._num_joints,
        )
        self._default_kp = self._as_vector(
            self._config.get("kps"),
            "kps",
            expected_size=self._num_joints,
        )
        self._default_kd = self._as_vector(
            self._config.get("kds"),
            "kds",
            expected_size=self._num_joints,
        )
        self._upper_body_config = UpperBodyConfig.from_config(self._config)

        xml_path = self._resolve_xml_path(str(bridge_config["xml_path"]))
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._simulation_dt
        self._torso_body_id = self._lookup_body_id("torso_link")

        imu_topic = str(self._config.get("imu_topic", DEFAULT_IMU_TOPIC))
        joint_state_topic = str(self._config.get("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC))

        self._imu_publisher = self.create_publisher(Imu, imu_topic, qos_profile_sensor_data)
        self._joint_state_publisher = self.create_publisher(
            JointStateArray,
            joint_state_topic,
            qos_profile_sensor_data,
        )
        self._group_runtimes = self._build_group_runtimes()
        self._leg_runtime = self._group_runtimes["leg"]

        self._apply_nominal_positions()
        mujoco.mj_forward(self._model, self._data)

        self._latest_tau_by_group = {
            group_name: np.zeros(len(runtime.spec.joint_order), dtype=np.float64)
            for group_name, runtime in self._group_runtimes.items()
        }
        self._publish_sequence = 0
        self._state_publish_accumulator = 0.0
        self._nominal_qpos = self._data.qpos.copy()
        self._nominal_qvel = np.zeros_like(self._data.qvel)
        self._viewer_context = None
        self._viewer = None
        self._viewer_sync_decimation = max(1, int(round(self._control_period_s / self._simulation_dt)))
        self._viewer_sync_counter = 0
        self._startup_hold_released = not self._hold_nominal_pose_until_first_command

        if enable_viewer:
            self._open_viewer()

        self._simulation_timer = self.create_timer(self._simulation_dt, self._simulation_step)
        self.get_logger().info(
            "MuJoCo HAL bridge ready. "
            f"xml={xml_path}, simulation_dt={self._simulation_dt:.4f}s, "
            f"state_publish_period_s={self._state_publish_period_s:.4f}s, "
            f"control_period_s={self._control_period_s:.4f}s, "
            f"hold_nominal_pose_until_first_command={self._hold_nominal_pose_until_first_command}, "
            f"groups={tuple(self._group_runtimes)}"
        )

    def _simulation_step(self) -> None:
        latest_commands = {
            group_name: runtime.adapter.get_latest_command()
            for group_name, runtime in self._group_runtimes.items()
        }
        if (
            self._hold_nominal_pose_until_first_command
            and not self._startup_hold_released
            and not any(command.has_received_command for command in latest_commands.values())
        ):
            self._hold_nominal_pose()
            return
        if (
            self._hold_nominal_pose_until_first_command
            and not self._startup_hold_released
            and any(command.has_received_command for command in latest_commands.values())
        ):
            self._startup_hold_released = True
            self.get_logger().info(
                "Received first joint command. Releasing nominal pose startup hold and resuming physics."
            )

        self._data.ctrl[:] = 0.0
        for group_name, runtime in self._group_runtimes.items():
            command = latest_commands[group_name]
            joint_position = self._data.qpos[runtime.qpos_indices]
            joint_velocity = self._data.qvel[runtime.qvel_indices]
            tau = command.effort + command.stiffness * (
                command.position - joint_position
            ) + command.damping * (command.velocity - joint_velocity)
            tau = np.clip(
                tau,
                runtime.actuator_ctrl_range[:, 0],
                runtime.actuator_ctrl_range[:, 1],
            )
            self._data.ctrl[runtime.actuator_indices] = tau
            self._latest_tau_by_group[group_name] = tau

        mujoco.mj_step(self._model, self._data)
        self._sync_viewer()

        self._state_publish_accumulator += self._simulation_dt
        if self._state_publish_accumulator + 1e-12 < self._state_publish_period_s:
            return

        self._publish_state()
        while self._state_publish_accumulator >= self._state_publish_period_s:
            self._state_publish_accumulator -= self._state_publish_period_s

    def _hold_nominal_pose(self) -> None:
        self._data.qpos[:] = self._nominal_qpos
        self._data.qvel[:] = self._nominal_qvel
        self._data.ctrl[:] = 0.0
        for tau in self._latest_tau_by_group.values():
            tau.fill(0.0)
        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()

        self._state_publish_accumulator += self._simulation_dt
        if self._state_publish_accumulator + 1e-12 < self._state_publish_period_s:
            return

        self._publish_state()
        while self._state_publish_accumulator >= self._state_publish_period_s:
            self._state_publish_accumulator -= self._state_publish_period_s

    def _publish_state(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        self._publish_imu(now_msg)
        self._publish_joint_state(now_msg)
        self._publish_sequence += 1

    def _publish_imu(self, now_msg) -> None:
        imu_message = Imu()
        imu_message.header.stamp = now_msg
        imu_message.header.frame_id = "torso_link"

        torso_quat_wxyz = self._data.xquat[self._torso_body_id]
        imu_message.orientation.x = float(torso_quat_wxyz[1])
        imu_message.orientation.y = float(torso_quat_wxyz[2])
        imu_message.orientation.z = float(torso_quat_wxyz[3])
        imu_message.orientation.w = float(torso_quat_wxyz[0])

        torso_angular_velocity = self._data.qvel[3:6]
        imu_message.angular_velocity.x = float(torso_angular_velocity[0])
        imu_message.angular_velocity.y = float(torso_angular_velocity[1])
        imu_message.angular_velocity.z = float(torso_angular_velocity[2])

        imu_message.orientation_covariance[0] = 0.0
        imu_message.angular_velocity_covariance[0] = 0.0
        imu_message.linear_acceleration_covariance[0] = -1.0

        self._imu_publisher.publish(imu_message)

    def _publish_joint_state(self, now_msg) -> None:
        joint_state_array = JointStateArray()
        joint_state_array.header.stamp = now_msg
        joint_state_array.header.meas_stamp = now_msg
        joint_state_array.header.sequence = self._publish_sequence
        joint_state_array.header.frame_id = "base"
        joint_state_array.state.value = DomainErrorState.NONE

        joint_position = self._data.qpos[self._leg_runtime.qpos_indices]
        joint_velocity = self._data.qvel[self._leg_runtime.qvel_indices]
        latest_tau = self._latest_tau_by_group["leg"]
        for index, joint_name in enumerate(self._leg_runtime.spec.joint_order):
            joint_state = JointState()
            joint_state.name = joint_name
            joint_state.position = float(joint_position[index])
            joint_state.velocity = float(joint_velocity[index])
            joint_state.effort = float(latest_tau[index])
            joint_state.coil_temp = 0
            joint_state.motor_temp = 0
            joint_state.motor_vol = 0
            joint_state_array.joints.append(joint_state)

        self._joint_state_publisher.publish(joint_state_array)

    def _build_group_runtimes(self) -> dict[str, JointGroupRuntime]:
        runtimes: dict[str, JointGroupRuntime] = {}
        for spec in self._build_group_specs():
            if not self._supports_joint_group(spec.joint_order):
                if spec.name == "leg":
                    raise ValueError(
                        f"Required leg joint group is not available in model: {spec.joint_order}"
                    )
                self.get_logger().warning(
                    f"Skipping unsupported joint group '{spec.name}' because the model does not contain all required joints/actuators."
                )
                continue
            qpos_indices, qvel_indices, actuator_indices, actuator_ctrl_range = self._build_joint_index_buffers(
                spec.joint_order
            )
            runtimes[spec.name] = JointGroupRuntime(
                spec=spec,
                qpos_indices=qpos_indices,
                qvel_indices=qvel_indices,
                actuator_indices=actuator_indices,
                actuator_ctrl_range=actuator_ctrl_range,
                adapter=MujocoCommandAdapter(
                    node=self,
                    topic_name=spec.topic_name,
                    joint_order=spec.joint_order,
                    default_position=spec.default_position,
                    default_velocity=np.zeros(len(spec.joint_order), dtype=np.float64),
                    default_effort=np.zeros(len(spec.joint_order), dtype=np.float64),
                    default_stiffness=spec.default_stiffness,
                    default_damping=spec.default_damping,
                ),
            )
        return runtimes

    def _build_group_specs(self) -> list[JointGroupSpec]:
        specs = [
            JointGroupSpec(
                name="leg",
                topic_name=str(self._config.get("joint_command_topic", DEFAULT_JOINT_COMMAND_TOPIC)),
                joint_order=self._leg_joint_order,
                default_position=self._default_angles.copy(),
                default_stiffness=self._default_kp.copy(),
                default_damping=self._default_kd.copy(),
            )
        ]
        if not self._upper_body_config.should_publish(is_mock_runtime=True):
            return specs

        upper_body_topics = self._config.get("upper_body_topics", {}) or {}
        if not isinstance(upper_body_topics, dict):
            raise ValueError("upper_body_topics must be a mapping when provided")
        specs.extend(
            [
                JointGroupSpec(
                    name="waist",
                    topic_name=str(upper_body_topics.get("waist", DEFAULT_WAIST_COMMAND_TOPIC)),
                    joint_order=tuple(WAIST_JOINT_ORDER),
                    default_position=self._upper_body_values(tuple(WAIST_JOINT_ORDER), self._upper_body_config.default_angles),
                    default_stiffness=self._upper_body_values(tuple(WAIST_JOINT_ORDER), self._upper_body_config.active_hold.kp),
                    default_damping=self._upper_body_values(tuple(WAIST_JOINT_ORDER), self._upper_body_config.active_hold.kd),
                ),
                JointGroupSpec(
                    name="arm",
                    topic_name=str(upper_body_topics.get("arm", DEFAULT_ARM_COMMAND_TOPIC)),
                    joint_order=tuple(ARM_JOINT_ORDER),
                    default_position=self._upper_body_values(tuple(ARM_JOINT_ORDER), self._upper_body_config.default_angles),
                    default_stiffness=self._upper_body_values(tuple(ARM_JOINT_ORDER), self._upper_body_config.active_hold.kp),
                    default_damping=self._upper_body_values(tuple(ARM_JOINT_ORDER), self._upper_body_config.active_hold.kd),
                ),
                JointGroupSpec(
                    name="head",
                    topic_name=str(upper_body_topics.get("head", DEFAULT_HEAD_COMMAND_TOPIC)),
                    joint_order=tuple(HEAD_JOINT_ORDER),
                    default_position=self._upper_body_values(tuple(HEAD_JOINT_ORDER), self._upper_body_config.default_angles),
                    default_stiffness=self._upper_body_values(tuple(HEAD_JOINT_ORDER), self._upper_body_config.active_hold.kp),
                    default_damping=self._upper_body_values(tuple(HEAD_JOINT_ORDER), self._upper_body_config.active_hold.kd),
                ),
            ]
        )
        return specs

    def _upper_body_values(self, joint_order: tuple[str, ...], values: np.ndarray) -> np.ndarray:
        upper_body_index = {
            joint_name: index for index, joint_name in enumerate(self._upper_body_config.joint_names)
        }
        return np.asarray([values[upper_body_index[joint_name]] for joint_name in joint_order], dtype=np.float64)

    def _supports_joint_group(self, joint_names: Sequence[str]) -> bool:
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"motor_{joint_name}",
            )
            if joint_id < 0 or actuator_id < 0:
                return False
        return True

    def _apply_nominal_positions(self) -> None:
        for runtime in self._group_runtimes.values():
            self._data.qpos[runtime.qpos_indices] = runtime.spec.default_position

    def _build_joint_index_buffers(
        self,
        joint_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        actuator_indices: list[int] = []
        actuator_ctrl_range: list[np.ndarray] = []

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' was not found in MuJoCo model")

            actuator_name = f"motor_{joint_name}"
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{actuator_name}' was not found in MuJoCo model")

            qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self._model.jnt_dofadr[joint_id]))
            actuator_indices.append(actuator_id)
            actuator_ctrl_range.append(self._model.actuator_ctrlrange[actuator_id].copy())

        return (
            np.asarray(qpos_indices, dtype=np.int32),
            np.asarray(qvel_indices, dtype=np.int32),
            np.asarray(actuator_indices, dtype=np.int32),
            np.asarray(actuator_ctrl_range, dtype=np.float64),
        )

    def _lookup_body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' was not found in MuJoCo model")
        return body_id

    def _open_viewer(self) -> None:
        self._viewer_context = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer = self._viewer_context.__enter__()
        self.get_logger().info("MuJoCo viewer enabled. Close the viewer window to stop visualization.")

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            self.get_logger().info("MuJoCo viewer window closed. Continuing bridge in headless mode.")
            self._close_viewer()
            return

        self._viewer_sync_counter += 1
        if self._viewer_sync_counter < self._viewer_sync_decimation:
            return

        self._viewer.sync()
        self._viewer_sync_counter = 0

    def _close_viewer(self) -> None:
        if self._viewer_context is not None:
            self._viewer_context.__exit__(None, None, None)
        self._viewer_context = None
        self._viewer = None
        self._viewer_sync_counter = 0

    def _resolve_xml_path(self, xml_path: str) -> Path:
        resolved = xml_path.replace("{LEGGED_GYM_ROOT_DIR}", str(self._project_root))
        candidate = Path(resolved)
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        return candidate.resolve()

    def _as_vector(
        self,
        values: Sequence[float] | None,
        field_name: str,
        *,
        expected_size: int,
    ) -> np.ndarray:
        if values is None:
            raise ValueError(f"Missing required config field '{field_name}'")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (expected_size,):
            raise ValueError(
                f"Expected config field '{field_name}' shape {(expected_size,)}, got {array.shape}"
            )
        return array.copy()


def _parse_args(args: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MuJoCo HAL mock bridge.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer window for live visualization.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = _parse_args(args)
    rclpy.init(args=args)
    node = MujocoHalBridge(config_path=parsed_args.config, enable_viewer=parsed_args.viewer)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_viewer()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
