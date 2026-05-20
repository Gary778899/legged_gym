from __future__ import annotations

import argparse
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
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMU_TOPIC,
    DEFAULT_JOINT_COMMAND_TOPIC,
    DEFAULT_JOINT_STATE_TOPIC,
    JOINT_ORDER,
)
from wbc_middleware.core.metadata_loader import load_yaml_config
from wbc_middleware.mock_bridge.mujoco_command_adapter import MujocoCommandAdapter


class MujocoHalBridge(Node):
    def __init__(self, config_path: str | None = None, *, enable_viewer: bool = False) -> None:
        config_path = config_path or DEFAULT_CONFIG_PATH
        self._config = load_yaml_config(config_path)
        bridge_config = dict(self._config.get("mock_bridge", {}))
        node_name = str(bridge_config.get("node_name", "mujoco_hal_bridge"))
        super().__init__(node_name)

        self._project_root = Path(__file__).resolve().parents[2]
        self._joint_order = tuple(JOINT_ORDER)
        self._num_joints = len(self._joint_order)

        self._simulation_dt = float(bridge_config["simulation_dt"])
        self._state_publish_period_s = float(bridge_config["state_publish_period_s"])
        self._control_period_s = float(bridge_config["control_period_s"])
        self._hold_nominal_pose_until_first_command = bool(
            bridge_config.get("hold_nominal_pose_until_first_command", False)
        )
        self._default_angles = self._as_vector(self._config.get("default_angles"), "default_angles")
        self._default_kp = self._as_vector(self._config.get("kps"), "kps")
        self._default_kd = self._as_vector(self._config.get("kds"), "kds")

        xml_path = self._resolve_xml_path(str(bridge_config["xml_path"]))
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._simulation_dt

        (
            self._qpos_indices,
            self._qvel_indices,
            self._actuator_indices,
            self._actuator_ctrl_range,
        ) = self._build_joint_index_buffers(self._joint_order)
        self._torso_body_id = self._lookup_body_id("torso_link")

        self._data.qpos[self._qpos_indices] = self._default_angles
        mujoco.mj_forward(self._model, self._data)

        self._latest_tau = np.zeros(self._num_joints, dtype=np.float64)
        self._publish_sequence = 0
        self._state_publish_accumulator = 0.0
        self._nominal_qpos = self._data.qpos.copy()
        self._nominal_qvel = np.zeros_like(self._data.qvel)
        self._viewer_context = None
        self._viewer = None
        self._viewer_sync_decimation = max(1, int(round(self._control_period_s / self._simulation_dt)))
        self._viewer_sync_counter = 0
        self._startup_hold_released = not self._hold_nominal_pose_until_first_command

        imu_topic = str(self._config.get("imu_topic", DEFAULT_IMU_TOPIC))
        joint_state_topic = str(self._config.get("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC))
        joint_command_topic = str(self._config.get("joint_command_topic", DEFAULT_JOINT_COMMAND_TOPIC))

        self._imu_publisher = self.create_publisher(Imu, imu_topic, qos_profile_sensor_data)
        self._joint_state_publisher = self.create_publisher(
            JointStateArray,
            joint_state_topic,
            qos_profile_sensor_data,
        )
        self._command_adapter = MujocoCommandAdapter(
            node=self,
            topic_name=joint_command_topic,
            joint_order=self._joint_order,
            default_position=self._default_angles,
            default_velocity=np.zeros(self._num_joints, dtype=np.float64),
            default_effort=np.zeros(self._num_joints, dtype=np.float64),
            default_stiffness=self._default_kp,
            default_damping=self._default_kd,
        )

        if enable_viewer:
            self._open_viewer()

        self._simulation_timer = self.create_timer(self._simulation_dt, self._simulation_step)
        self.get_logger().info(
            "MuJoCo HAL bridge ready. "
            f"xml={xml_path}, simulation_dt={self._simulation_dt:.4f}s, "
            f"state_publish_period_s={self._state_publish_period_s:.4f}s, "
            f"control_period_s={self._control_period_s:.4f}s, "
            f"hold_nominal_pose_until_first_command={self._hold_nominal_pose_until_first_command}"
        )

    def _simulation_step(self) -> None:
        command = self._command_adapter.get_latest_command()
        if (
            self._hold_nominal_pose_until_first_command
            and not self._startup_hold_released
            and not command.has_received_command
        ):
            self._hold_nominal_pose()
            return
        if (
            self._hold_nominal_pose_until_first_command
            and not self._startup_hold_released
            and command.has_received_command
        ):
            self._startup_hold_released = True
            self.get_logger().info(
                "Received first joint command. Releasing nominal pose startup hold and resuming physics."
            )

        joint_position = self._data.qpos[self._qpos_indices]
        joint_velocity = self._data.qvel[self._qvel_indices]

        tau = command.effort + command.stiffness * (command.position - joint_position) + command.damping * (
            command.velocity - joint_velocity
        )
        tau = np.clip(tau, self._actuator_ctrl_range[:, 0], self._actuator_ctrl_range[:, 1])
        self._data.ctrl[self._actuator_indices] = tau
        self._latest_tau = tau

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
        self._latest_tau.fill(0.0)
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

        joint_position = self._data.qpos[self._qpos_indices]
        joint_velocity = self._data.qvel[self._qvel_indices]
        for index, joint_name in enumerate(self._joint_order):
            joint_state = JointState()
            joint_state.name = joint_name
            joint_state.position = float(joint_position[index])
            joint_state.velocity = float(joint_velocity[index])
            joint_state.effort = float(self._latest_tau[index])
            joint_state.coil_temp = 0
            joint_state.motor_temp = 0
            joint_state.motor_vol = 0
            joint_state_array.joints.append(joint_state)

        self._joint_state_publisher.publish(joint_state_array)

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

    def _as_vector(self, values: Sequence[float] | None, field_name: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"Missing required config field '{field_name}'")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._num_joints,):
            raise ValueError(
                f"Expected config field '{field_name}' shape {(self._num_joints,)}, got {array.shape}"
            )
        return array


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
