from __future__ import annotations

import csv
from enum import Enum
from math import asin, atan2
from pathlib import Path

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from aimdk_msgs.msg import JointStateArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from wbc_middleware.core.command_mapper import CommandMapper, SafetyLimits
from wbc_middleware.core.command_sources import build_command_source
from wbc_middleware.core.constants import (
    CONTROL_PERIOD_S,
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HEAD_COMMAND_TOPIC,
    DEFAULT_IMU_TOPIC,
    DEFAULT_INTERACTIVE_COMMAND_TOPIC,
    DEFAULT_JOINT_COMMAND_TOPIC,
    DEFAULT_JOINT_STATE_TOPIC,
    DEFAULT_STARTUP_TRIGGER_SERVICE,
    DEFAULT_STOP_TRIGGER_SERVICE,
    DEFAULT_WAIST_COMMAND_TOPIC,
    JOINT_NAME_TO_INDEX,
)
from wbc_middleware.core.metadata_loader import load_runtime_metadata, load_yaml_config
from wbc_middleware.core.observation_builder import ObservationBuilder
from wbc_middleware.core.onnx_policy_runner import OnnxPolicyRunner
from wbc_middleware.core.robot_state import RobotState
from wbc_middleware.core.upper_body_command_builder import (
    UpperBodyCommandBuilder,
    split_upper_body_targets_by_area,
)
from wbc_middleware.core.upper_body_config import UpperBodyConfig
from wbc_middleware.ros2.hal_command_publisher import HalCommandPublisher




class FallLogger:
    FIELDNAMES = (
        'time_s',
        'startup_state',
        'command_x',
        'command_y',
        'command_yaw',
        'torso_roll',
        'torso_pitch',
        'torso_roll_rate',
        'torso_pitch_rate',
        'action_norm',
        'action_max_abs',
        'left_hip_pitch_target',
        'right_hip_pitch_target',
        'left_knee_target',
        'right_knee_target',
        'left_ankle_pitch_target',
        'right_ankle_pitch_target',
        'waist_yaw_target',
        'waist_pitch_target',
        'waist_roll_target',
        'left_elbow_target',
        'right_elbow_target',
        'head_pitch_target',
    )

    def __init__(self, enabled: bool, path: str):
        self._enabled = enabled
        self._path = path
        self._file = None
        self._writer = None
        if not self._enabled:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open('w', encoding='utf-8', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> str:
        return self._path

    def log_row(self, row: dict[str, float | str]) -> None:
        if not self._enabled or self._writer is None or self._file is None:
            return
        serialized = {field: row.get(field, '') for field in self.FIELDNAMES}
        self._writer.writerow(serialized)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

class StartupState(str, Enum):
    IDLE = 'IDLE'
    MOVE_TO_DEFAULT = 'MOVE_TO_DEFAULT'
    POLICY_HOLD = 'POLICY_HOLD'
    POLICY_ACTIVE = 'POLICY_ACTIVE'
    STOPPING = 'STOPPING'
    SAFE_HOLD = 'SAFE_HOLD'


class ControlNode(Node):
    def __init__(self, config_path: str | None = None):
        config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = load_yaml_config(config_path)
        self.config['policy_path'] = str(
            self._resolve_project_path(self.config.get('policy_path'))
        )
        self.config['metadata_path'] = str(
            self._resolve_project_path(self.config.get('metadata_path'))
        )
        node_name = str(self.config.get('node_name', 'control_middleware'))
        super().__init__(node_name)

        self.state = RobotState()
        self.default_dof_pos = np.asarray(
            self.config.get('default_angles', self.state.default_dof_pos()),
            dtype=np.float64,
        )
        self.state.set_default_dof_pos(self.default_dof_pos)

        self.runtime_metadata = load_runtime_metadata(
            policy_path=self.config.get('policy_path'),
            metadata_path=self.config.get('metadata_path'),
            yaml_config=self.config,
        )
        self.scales = self.runtime_metadata.build_scaling_values()
        self.observation_builder = ObservationBuilder(
            default_dof_pos=self.default_dof_pos,
            scales=self.scales,
            clip_observations=self.runtime_metadata.clip_observations,
        )
        safety_config = self.config.get('safety', {})
        self.command_mapper = CommandMapper(
            default_dof_pos=self.default_dof_pos,
            metadata=self.runtime_metadata,
            safety_limits=SafetyLimits.from_config(
                safety_config,
                num_joints=len(self.default_dof_pos),
                default_action_clip=self.runtime_metadata.clip_actions,
                default_position_delta_clip=(
                    self.runtime_metadata.action_scale * self.runtime_metadata.clip_actions
                ),
            ),
        )
        self.command_source = build_command_source(self.config.get('command_source'))
        self.leg_command_publisher = HalCommandPublisher(
            self,
            str(self.config.get('joint_command_topic', DEFAULT_JOINT_COMMAND_TOPIC)),
        )
        self.is_mock_runtime = self._resolve_is_mock_runtime(self.config)
        self.upper_body_config = UpperBodyConfig.from_config(self.config)
        self.upper_body_builder = UpperBodyCommandBuilder(self.upper_body_config)
        self.upper_body_publish_enabled = self.upper_body_config.should_publish(
            is_mock_runtime=self.is_mock_runtime
        )
        self.upper_body_publishers = self._build_upper_body_publishers()
        self.policy_runner = OnnxPolicyRunner(
            model_path=str(self.config.get('policy_path')),
            hidden_dim=self.runtime_metadata.rnn_hidden_size,
            num_layers=self.runtime_metadata.rnn_num_layers,
            num_envs=1,
        )

        # Match training: phase is a normalized gait clock in [0, 1),
        # then the observation uses sin(2*pi*phase), cos(2*pi*phase).
        self.phase = 0.0
        self.print_counter = 0
        self.control_period_s = float(self.config.get('control_period_s', CONTROL_PERIOD_S))
        self.state_timeout_s = float(self.config.get('state_timeout_s', 0.25))
        startup_config = self.config.get('startup', {})
        self.move_to_default_tolerance = float(
            startup_config.get('move_to_default_tolerance', 0.15)
        )
        self.policy_hold_time_s = float(startup_config.get('policy_hold_time_s', 0.5))
        self.stopping_hold_time_s = float(startup_config.get('stopping_hold_time_s', 0.5))
        self.interactive_command_timeout_s = float(
            self.config.get('interactive_command_timeout_s', 0.5)
        )
        self.publish_safe_hold_on_shutdown = bool(
            safety_config.get('publish_safe_hold_on_shutdown', True)
        )
        self.safe_hold_mode = self._resolve_safe_hold_mode(safety_config)
        self.safe_hold_damping = self._resolve_safe_hold_damping(safety_config)
        self.imu_ready = False
        self.joints_ready = False
        self.startup_state = StartupState.IDLE
        self.policy_hold_started_at_s: float | None = None
        self.stopping_started_at_s: float | None = None
        self.interactive_command: np.ndarray | None = None
        self.interactive_command_stamp_s: float | None = None
        self._state_timeout_reported = False
        self._state_ready_once = False
        self._shutdown_requested = False
        self._fall_logger = self._build_fall_logger()

        imu_topic = str(self.config.get('imu_topic', DEFAULT_IMU_TOPIC))
        joint_state_topic = str(self.config.get('joint_state_topic', DEFAULT_JOINT_STATE_TOPIC))
        interactive_command_topic = str(
            self.config.get('interactive_command_topic', DEFAULT_INTERACTIVE_COMMAND_TOPIC)
        )
        startup_service_name = str(
            self.config.get('startup_trigger_service', DEFAULT_STARTUP_TRIGGER_SERVICE)
        )
        stop_service_name = str(
            self.config.get('stop_trigger_service', DEFAULT_STOP_TRIGGER_SERVICE)
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
        self.stop_service = self.create_service(
            Trigger,
            stop_service_name,
            self._handle_stop_trigger,
        )
        self.timer = self.create_timer(self.control_period_s, self._control_loop)
        self.get_logger().info(
            f'Middleware started with config {Path(config_path)}. '
            f'Startup state={self.startup_state.value}. Waiting for HAL state.'
        )
        if self.upper_body_config.enabled:
            self.get_logger().info(
                'Upper-body stabilization configured with '
                f'{len(self.upper_body_config.joint_names)} joints; '
                f'publish_enabled={self.upper_body_publish_enabled}, '
                f'is_mock_runtime={self.is_mock_runtime}.'
            )
        if self._fall_logger.enabled:
            self.get_logger().info(f'Fall logger enabled. Writing CSV to {self._fall_logger.path}')

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
        self._state_timeout_reported = False
        self._state_ready_once = self._state_ready_once or self.joints_ready

    def joint_state_callback(self, msg: JointStateArray) -> None:
        for joint_data in msg.joints:
            if joint_data.name in JOINT_NAME_TO_INDEX:
                index = JOINT_NAME_TO_INDEX[joint_data.name]
                self.state.joint_pos[index] = joint_data.position
                self.state.joint_vel[index] = joint_data.velocity
        self.state.joint_stamp_s = self._clock_now_s()
        self.joints_ready = True
        self._state_timeout_reported = False
        self._state_ready_once = self._state_ready_once or self.imu_ready

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
        self.get_logger().info('LSTM memory reset to zeros.')

    def publish_default_pose(self) -> None:
        self._publish_control_targets(
            self.command_mapper.build_default_pose_targets(),
            upper_body_mode='active_hold',
        )

    def run_policy_step(self) -> np.ndarray:
        gait_period_s = 1.0
        self.phase = (self.phase + self.control_period_s / gait_period_s) % 1.0
        observations = self.build_observations()
        action = self.policy_runner.run(observations)
        mapped_action = self.command_mapper.map_action(action)
        self.state.last_action = mapped_action.safe_action
        targets = mapped_action.targets
        self._publish_control_targets(targets, upper_body_mode='active_hold')

        self.print_counter += 1
        if self.print_counter >= int(round(1.0 / self.control_period_s)):
            formatted_obs = ', '.join(f'{value:.3f}' for value in observations.flatten())
            self.get_logger().info(
                f'\n--- FULL OBSERVATION ({observations.shape[1]} Dims) ---\n'
                f'{formatted_obs}\n----------------------------------'
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
                'Startup sequence already active; only IDLE can transition to MOVE_TO_DEFAULT '
                f'(current={self.startup_state.value}).'
            )
            return response
        self._transition_to(StartupState.MOVE_TO_DEFAULT)
        response.success = True
        response.message = 'Startup sequence accepted: IDLE -> MOVE_TO_DEFAULT.'
        return response

    def _handle_stop_trigger(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self.startup_state in (StartupState.IDLE, StartupState.SAFE_HOLD):
            response.success = False
            response.message = (
                f'Stop request ignored because middleware is already in {self.startup_state.value}.'
            )
            return response
        self.request_safe_stop('stop service requested')
        response.success = True
        response.message = f'Safe stop accepted: entering {self.startup_state.value}.'
        return response

    def _transition_to(self, next_state: StartupState) -> None:
        previous_state = self.startup_state
        self.startup_state = next_state
        if next_state is StartupState.IDLE:
            self.policy_hold_started_at_s = None
            self.stopping_started_at_s = None
        elif next_state is StartupState.MOVE_TO_DEFAULT:
            self.policy_hold_started_at_s = None
            self.stopping_started_at_s = None
        elif next_state is StartupState.POLICY_HOLD:
            self.reset_lstm_memory()
            self.policy_hold_started_at_s = self._clock_now_s()
            self.stopping_started_at_s = None
        elif next_state is StartupState.POLICY_ACTIVE:
            self.policy_hold_started_at_s = None
            self.stopping_started_at_s = None
            self.command_source.reset()
        elif next_state is StartupState.STOPPING:
            self.policy_hold_started_at_s = None
            self.stopping_started_at_s = self._clock_now_s()
            self.interactive_command = None
            self.interactive_command_stamp_s = None
            self.state.last_action.fill(0.0)
        elif next_state is StartupState.SAFE_HOLD:
            self.policy_hold_started_at_s = None
            self.stopping_started_at_s = None
            self.interactive_command = None
            self.interactive_command_stamp_s = None
            self.state.last_action.fill(0.0)
            self.reset_lstm_memory()
        self.get_logger().info(
            f'Startup state transition: {previous_state.value} -> {next_state.value}'
        )

    def _max_default_pose_error(self) -> float:
        return float(np.max(np.abs(self.state.joint_pos - self.default_dof_pos)))

    def get_active_command(self) -> np.ndarray:
        if self.startup_state is not StartupState.POLICY_ACTIVE:
            return np.zeros(3, dtype=np.float32)
        if (
            self.interactive_command is not None
            and self.interactive_command_stamp_s is not None
            and (self._clock_now_s() - self.interactive_command_stamp_s)
            <= self.interactive_command_timeout_s
        ):
            return self.interactive_command.copy()
        return self.command_source.get_command()

    def request_safe_stop(self, reason: str) -> None:
        if self.startup_state in (StartupState.IDLE, StartupState.STOPPING, StartupState.SAFE_HOLD):
            return
        self._log_warning(f'Requesting safe stop: {reason}')
        self._transition_to(StartupState.STOPPING)

    def publish_safe_hold(self, reason: str | None = None) -> None:
        if not self._context_is_ok():
            if reason:
                print(f'[control_middleware] Skipping safe-hold publish because ROS context is no longer valid: {reason}')
            return
        if reason:
            self._log_warning(f'Publishing safe hold command: {reason}')
        self._publish_control_targets(
            self._build_lower_body_safe_hold_targets(),
            upper_body_mode='safe_hold',
        )

    def _handle_state_timeout(self) -> None:
        if self._state_timeout_reported:
            return
        self._state_timeout_reported = True
        self._log_warning(
            f'HAL state timed out. Entering safe stop path with safe_hold_mode={self.safe_hold_mode}.'
        )
        if self.startup_state not in (StartupState.IDLE, StartupState.SAFE_HOLD):
            self.request_safe_stop('HAL state timeout')

    def _control_loop(self) -> None:
        has_fresh_state = self.has_fresh_state()
        if not has_fresh_state:
            if self._state_ready_once:
                self._handle_state_timeout()
                if self.startup_state in (StartupState.STOPPING, StartupState.SAFE_HOLD):
                    self.publish_safe_hold()
            return

        self._state_timeout_reported = False
        self._state_ready_once = True

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
            return

        if self.startup_state is StartupState.STOPPING:
            self.publish_safe_hold()
            if self.stopping_started_at_s is None:
                self.stopping_started_at_s = self._clock_now_s()
            if (self._clock_now_s() - self.stopping_started_at_s) >= self.stopping_hold_time_s:
                self._transition_to(StartupState.SAFE_HOLD)
            return

        if self.startup_state is StartupState.SAFE_HOLD:
            self.publish_safe_hold()

    def shutdown_to_safe_hold(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        if not self.publish_safe_hold_on_shutdown:
            return
        if not self._context_is_ok():
            print('[control_middleware] ROS context already invalid during shutdown; skipping safe-hold publish.')
            return
        try:
            if self.startup_state not in (StartupState.IDLE, StartupState.SAFE_HOLD):
                self._transition_to(StartupState.SAFE_HOLD)
            self.publish_safe_hold('node shutdown')
        except Exception as exc:
            self._log_warning(f'Failed to publish shutdown safe-hold command: {exc}')
        finally:
            self._fall_logger.close()

    def _publish_control_targets(
        self,
        lower_body_targets,
        *,
        upper_body_mode: str | None = None,
    ) -> None:
        self.leg_command_publisher.publish(lower_body_targets)
        upper_body_targets = self._publish_upper_body_targets(upper_body_mode)
        self._log_fall_frame(lower_body_targets, upper_body_targets)

    def _publish_upper_body_targets(self, mode: str | None) -> list | None:
        if not self.upper_body_publish_enabled or mode is None:
            return None
        if mode == 'active_hold':
            torso_roll, torso_pitch = self._torso_roll_pitch()
            targets = self.upper_body_builder.build_active_hold_targets(
                torso_roll=torso_roll,
                torso_pitch=torso_pitch,
                torso_roll_rate=float(self.state.gyro[0]),
                torso_pitch_rate=float(self.state.gyro[1]),
            )
        elif mode == 'safe_hold':
            targets = self.upper_body_builder.build_safe_hold_targets()
        else:
            raise ValueError(f'Unsupported upper-body mode: {mode}')
        grouped_targets = split_upper_body_targets_by_area(targets)
        for area_name, area_targets in grouped_targets.items():
            publisher = self.upper_body_publishers.get(area_name)
            if publisher is None:
                continue
            publisher.publish(area_targets)
        return targets

    def _build_lower_body_safe_hold_targets(self):
        if self.safe_hold_mode == 'damping':
            return self.command_mapper.build_damping_targets(
                damping=self.safe_hold_damping,
                position_reference=self.state.joint_pos,
            )
        return self.command_mapper.build_default_pose_targets()

    def _build_upper_body_publishers(self) -> dict[str, HalCommandPublisher]:
        if not self.upper_body_publish_enabled:
            return {}
        upper_body_topics = self.config.get('upper_body_topics', {}) or {}
        if not isinstance(upper_body_topics, dict):
            raise ValueError('upper_body_topics must be a mapping when provided')
        return {
            'waist': HalCommandPublisher(
                self,
                str(upper_body_topics.get('waist', DEFAULT_WAIST_COMMAND_TOPIC)),
            ),
            'arm': HalCommandPublisher(
                self,
                str(upper_body_topics.get('arm', DEFAULT_ARM_COMMAND_TOPIC)),
            ),
            'head': HalCommandPublisher(
                self,
                str(upper_body_topics.get('head', DEFAULT_HEAD_COMMAND_TOPIC)),
            ),
        }

    def _resolve_safe_hold_mode(self, safety_config: dict) -> str:
        safe_hold_mode = str(safety_config.get('safe_hold_mode', 'hold_position')).lower()
        if safe_hold_mode not in {'hold_position', 'damping'}:
            raise ValueError(
                'safety.safe_hold_mode must be one of: hold_position, damping'
            )
        return safe_hold_mode

    def _resolve_safe_hold_damping(self, safety_config: dict) -> np.ndarray:
        damping_value = safety_config.get('safe_hold_damping')
        if damping_value is None:
            return self.runtime_metadata.kd.copy()
        damping_array = np.asarray(damping_value, dtype=np.float64)
        if damping_array.shape != self.runtime_metadata.kd.shape:
            raise ValueError(
                f'Expected safe_hold_damping shape {self.runtime_metadata.kd.shape}, got {damping_array.shape}'
            )
        return damping_array.copy()

    def _torso_roll_pitch(self) -> tuple[float, float]:
        x, y, z, w = (float(value) for value in self.state.quat)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = asin(sinp)
        return roll, pitch


    def _build_fall_logger(self) -> FallLogger:
        diagnostics = self.config.get('diagnostics', {}) or {}
        if not isinstance(diagnostics, dict):
            raise ValueError('diagnostics must be a mapping when provided')
        fall_logger_config = diagnostics.get('fall_logger', {}) or {}
        if not isinstance(fall_logger_config, dict):
            raise ValueError('diagnostics.fall_logger must be a mapping when provided')
        enabled = bool(fall_logger_config.get('enabled', True))
        configured_path = str(fall_logger_config.get('path', 'logs/csv/control_middleware_fall_log.csv'))
        output_path = str(self._resolve_project_path(configured_path))
        return FallLogger(enabled=enabled, path=output_path)

    def _log_fall_frame(self, lower_body_targets, upper_body_targets) -> None:
        if not self._fall_logger.enabled:
            return
        lower_targets = {target.name: float(target.position) for target in lower_body_targets}
        upper_targets = {target.name: float(target.position) for target in (upper_body_targets or [])}
        torso_roll, torso_pitch = self._torso_roll_pitch()
        active_command = self.get_active_command()
        action_norm = float(np.linalg.norm(self.state.last_action))
        action_max_abs = float(np.max(np.abs(self.state.last_action))) if self.state.last_action.size else 0.0
        row = {
            'time_s': self._clock_now_s(),
            'startup_state': self.startup_state.value,
            'command_x': float(active_command[0]),
            'command_y': float(active_command[1]),
            'command_yaw': float(active_command[2]),
            'torso_roll': torso_roll,
            'torso_pitch': torso_pitch,
            'torso_roll_rate': float(self.state.gyro[0]),
            'torso_pitch_rate': float(self.state.gyro[1]),
            'action_norm': action_norm,
            'action_max_abs': action_max_abs,
            'left_hip_pitch_target': lower_targets.get('left_hip_pitch_joint', ''),
            'right_hip_pitch_target': lower_targets.get('right_hip_pitch_joint', ''),
            'left_knee_target': lower_targets.get('left_knee_joint', ''),
            'right_knee_target': lower_targets.get('right_knee_joint', ''),
            'left_ankle_pitch_target': lower_targets.get('left_ankle_pitch_joint', ''),
            'right_ankle_pitch_target': lower_targets.get('right_ankle_pitch_joint', ''),
            'waist_yaw_target': upper_targets.get('waist_yaw_joint', ''),
            'waist_pitch_target': upper_targets.get('waist_pitch_joint', ''),
            'waist_roll_target': upper_targets.get('waist_roll_joint', ''),
            'left_elbow_target': upper_targets.get('left_elbow_joint', ''),
            'right_elbow_target': upper_targets.get('right_elbow_joint', ''),
            'head_pitch_target': upper_targets.get('head_pitch_joint', ''),
        }
        self._fall_logger.log_row(row)

    def _resolve_is_mock_runtime(self, config: dict) -> bool:
        if 'is_mock_runtime' in config:
            return bool(config.get('is_mock_runtime'))
        return bool(config.get('mock_bridge'))

    def _context_is_ok(self) -> bool:
        try:
            return bool(rclpy.ok(context=self.context))
        except Exception:
            return False

    def _log_warning(self, message: str) -> None:
        if self._context_is_ok():
            self.get_logger().warning(message)
        else:
            print(f'[control_middleware][WARN] {message}')

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
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_to_safe_hold()
        node.destroy_node()
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)
