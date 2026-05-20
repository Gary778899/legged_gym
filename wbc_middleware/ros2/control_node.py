from __future__ import annotations

from enum import Enum
from pathlib import Path

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from aimdk_msgs.msg import JointStateArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from wbc_middleware.core.command_mapper import CommandMapper
from wbc_middleware.core.command_sources import build_command_source
from wbc_middleware.core.constants import (
    CONTROL_PERIOD_S,
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMU_TOPIC,
    DEFAULT_INTERACTIVE_COMMAND_TOPIC,
    DEFAULT_JOINT_COMMAND_TOPIC,
    DEFAULT_JOINT_STATE_TOPIC,
    DEFAULT_STARTUP_TRIGGER_SERVICE,
    JOINT_NAME_TO_INDEX,
)
from wbc_middleware.core.metadata_loader import load_runtime_metadata, load_yaml_config
from wbc_middleware.core.observation_builder import ObservationBuilder
from wbc_middleware.core.onnx_policy_runner import OnnxPolicyRunner
from wbc_middleware.core.robot_state import RobotState
from wbc_middleware.ros2.hal_command_publisher import HalCommandPublisher


class StartupState(str, Enum):
    IDLE = "IDLE"
    MOVE_TO_DEFAULT = "MOVE_TO_DEFAULT"
    POLICY_HOLD = "POLICY_HOLD"
    POLICY_ACTIVE = "POLICY_ACTIVE"


class ControlNode(Node):
    def __init__(self, config_path: str | None = None):
        config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = load_yaml_config(config_path)
        self.config["policy_path"] = str(
            self._resolve_project_path(self.config.get("policy_path"))
        )
        self.config["metadata_path"] = str(
            self._resolve_project_path(self.config.get("metadata_path"))
        )
        node_name = str(self.config.get("node_name", "control_middleware"))
        super().__init__(node_name)

        self.state = RobotState()
        self.default_dof_pos = np.asarray(
            self.config.get("default_angles", self.state.default_dof_pos()),
            dtype=np.float64,
        )
        self.state.set_default_dof_pos(self.default_dof_pos)

        self.runtime_metadata = load_runtime_metadata(
            policy_path=self.config.get("policy_path"),
            metadata_path=self.config.get("metadata_path"),
            yaml_config=self.config,
        )
        self.scales = self.runtime_metadata.build_scaling_values()
        self.observation_builder = ObservationBuilder(
            default_dof_pos=self.default_dof_pos,
            scales=self.scales,
        )
        self.command_mapper = CommandMapper(
            default_dof_pos=self.default_dof_pos,
            metadata=self.runtime_metadata,
        )
        self.command_source = build_command_source(self.config.get("command_source"))
        self.command_publisher = HalCommandPublisher(
            self,
            str(self.config.get("joint_command_topic", DEFAULT_JOINT_COMMAND_TOPIC)),
        )
        self.policy_runner = OnnxPolicyRunner(
            model_path=str(self.config.get("policy_path")),
            hidden_dim=self.runtime_metadata.rnn_hidden_size,
            num_layers=self.runtime_metadata.rnn_num_layers,
            num_envs=1,
        )

        self.phase = 0.0
        self.print_counter = 0
        self.control_period_s = float(self.config.get("control_period_s", CONTROL_PERIOD_S))
        self.state_timeout_s = float(self.config.get("state_timeout_s", 0.25))
        startup_config = self.config.get("startup", {})
        self.move_to_default_tolerance = float(
            startup_config.get("move_to_default_tolerance", 0.15)
        )
        self.policy_hold_time_s = float(startup_config.get("policy_hold_time_s", 0.5))
        self.interactive_command_timeout_s = float(
            self.config.get("interactive_command_timeout_s", 0.5)
        )
        self.imu_ready = False
        self.joints_ready = False
        self.startup_state = StartupState.IDLE
        self.policy_hold_started_at_s: float | None = None
        self.interactive_command: np.ndarray | None = None
        self.interactive_command_stamp_s: float | None = None

        imu_topic = str(self.config.get("imu_topic", DEFAULT_IMU_TOPIC))
        joint_state_topic = str(self.config.get("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC))
        interactive_command_topic = str(
            self.config.get("interactive_command_topic", DEFAULT_INTERACTIVE_COMMAND_TOPIC)
        )
        startup_service_name = str(
            self.config.get("startup_trigger_service", DEFAULT_STARTUP_TRIGGER_SERVICE)
        )
        self.create_subscription(Imu, imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(
            JointStateArray,
            joint_state_topic,
            self.joint_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            interactive_command_topic,
            self.interactive_command_callback,
            10,
        )
        self.startup_service = self.create_service(
            Trigger,
            startup_service_name,
            self._handle_startup_trigger,
        )
        self.timer = self.create_timer(self.control_period_s, self._control_loop)
        self.get_logger().info(
            f"Middleware started with config {Path(config_path)}. "
            f"Startup state={self.startup_state.value}. Waiting for HAL state."
        )

    def imu_callback(self, msg: Imu) -> None:
        self.state.quat = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=np.float64,
        )
        self.state.gyro = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        self.state.imu_stamp_s = self._clock_now_s()
        self.imu_ready = True

    def joint_state_callback(self, msg: JointStateArray) -> None:
        for joint_data in msg.joints:
            if joint_data.name in JOINT_NAME_TO_INDEX:
                index = JOINT_NAME_TO_INDEX[joint_data.name]
                self.state.joint_pos[index] = joint_data.position
                self.state.joint_vel[index] = joint_data.velocity
        self.state.joint_stamp_s = self._clock_now_s()
        self.joints_ready = True

    def interactive_command_callback(self, msg: Twist) -> None:
        self.interactive_command = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z],
            dtype=np.float32,
        )
        self.interactive_command_stamp_s = self._clock_now_s()

    def build_observations(self) -> np.ndarray:
        return self.observation_builder.build(
            state=self.state,
            commands=self.get_active_command(),
            phase=self.phase,
        )

    def reset_lstm_memory(self) -> None:
        self.policy_runner.reset_memory()
        self.state.last_action.fill(0.0)
        self.get_logger().info("LSTM memory reset to zeros.")

    def publish_default_pose(self) -> None:
        self.command_publisher.publish(self.command_mapper.build_default_pose_targets())

    def run_policy_step(self) -> np.ndarray:
        self.phase = (self.phase + self.control_period_s) % (2 * np.pi)
        observations = self.build_observations()
        action = self.policy_runner.run(observations)
        self.state.last_action = action
        targets = self.command_mapper.map_action_to_targets(action)
        self.command_publisher.publish(targets)

        self.print_counter += 1
        if self.print_counter >= int(round(1.0 / self.control_period_s)):
            formatted_obs = ", ".join(f"{value:.3f}" for value in observations.flatten())
            self.get_logger().info(
                f"\n--- FULL OBSERVATION ({observations.shape[1]} Dims) ---\n"
                f"{formatted_obs}\n----------------------------------"
            )
            self.print_counter = 0

        return action

    def has_fresh_state(self) -> bool:
        if not (self.imu_ready and self.joints_ready):
            return False
        now_s = self._clock_now_s()
        if now_s - self.state.imu_stamp_s > self.state_timeout_s:
            return False
        if now_s - self.state.joint_stamp_s > self.state_timeout_s:
            return False
        return True

    def _handle_startup_trigger(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self.startup_state is not StartupState.IDLE:
            response.success = False
            response.message = (
                "Startup sequence already active; only IDLE can transition to MOVE_TO_DEFAULT "
                f"(current={self.startup_state.value})."
            )
            return response
        self._transition_to(StartupState.MOVE_TO_DEFAULT)
        response.success = True
        response.message = "Startup sequence accepted: IDLE -> MOVE_TO_DEFAULT."
        return response

    def _transition_to(self, next_state: StartupState) -> None:
        previous_state = self.startup_state
        self.startup_state = next_state
        if next_state is StartupState.IDLE:
            self.policy_hold_started_at_s = None
        elif next_state is StartupState.MOVE_TO_DEFAULT:
            self.policy_hold_started_at_s = None
        elif next_state is StartupState.POLICY_HOLD:
            self.reset_lstm_memory()
            self.policy_hold_started_at_s = self._clock_now_s()
        elif next_state is StartupState.POLICY_ACTIVE:
            self.policy_hold_started_at_s = None
            self.command_source.reset()
        self.get_logger().info(
            f"Startup state transition: {previous_state.value} -> {next_state.value}"
        )

    def _max_default_pose_error(self) -> float:
        return float(np.max(np.abs(self.state.joint_pos - self.default_dof_pos)))

    def get_active_command(self) -> np.ndarray:
        if (
            self.interactive_command is not None
            and self.interactive_command_stamp_s is not None
            and (self._clock_now_s() - self.interactive_command_stamp_s)
            <= self.interactive_command_timeout_s
        ):
            return self.interactive_command.copy()
        return self.command_source.get_command()

    def _control_loop(self) -> None:
        if not self.has_fresh_state():
            return
        if self.startup_state is StartupState.IDLE:
            return

        if self.startup_state is StartupState.MOVE_TO_DEFAULT:
            self.publish_default_pose()
            if self._max_default_pose_error() < self.move_to_default_tolerance:
                self._transition_to(StartupState.POLICY_HOLD)
            return

        if self.startup_state is StartupState.POLICY_HOLD:
            self.publish_default_pose()
            if self.policy_hold_started_at_s is None:
                self.policy_hold_started_at_s = self._clock_now_s()
            if (self._clock_now_s() - self.policy_hold_started_at_s) >= self.policy_hold_time_s:
                self._transition_to(StartupState.POLICY_ACTIVE)
            return

        if self.startup_state is StartupState.POLICY_ACTIVE:
            self.run_policy_step()

    def _clock_now_s(self) -> float:
        now = self.get_clock().now()
        return float(now.nanoseconds) * 1e-9

    def _resolve_project_path(self, path_value: str | None) -> Path:
        if not path_value:
            return Path.cwd()
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (Path(__file__).resolve().parents[2] / candidate).resolve()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
