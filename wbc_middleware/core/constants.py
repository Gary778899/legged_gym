JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

WAIST_JOINT_ORDER = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]

ARM_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

HEAD_JOINT_ORDER = [
    "head_yaw_joint",
    "head_pitch_joint",
]

UPPER_BODY_JOINT_ORDER = WAIST_JOINT_ORDER + ARM_JOINT_ORDER + HEAD_JOINT_ORDER

NUM_ACTIONS = 12
NUM_OBS = 47

DEFAULT_CONFIG_PATH = "wbc_middleware/config/x2_middleware.yaml"
DEFAULT_POLICY_PATH = "wbc_middleware/policy.onnx"
DEFAULT_METADATA_PATH = "wbc_middleware/onnx_metadata.json"

DEFAULT_IMU_TOPIC = "/aima/hal/imu/torso/state"
DEFAULT_JOINT_STATE_TOPIC = "/aima/hal/joint/leg/state"
DEFAULT_JOINT_COMMAND_TOPIC = "/aima/hal/joint/leg/command"
DEFAULT_WAIST_COMMAND_TOPIC = "/aima/hal/joint/waist/command"
DEFAULT_ARM_COMMAND_TOPIC = "/aima/hal/joint/arm/command"
DEFAULT_HEAD_COMMAND_TOPIC = "/aima/hal/joint/head/command"
DEFAULT_STARTUP_TRIGGER_SERVICE = "/aima/middleware/startup"
DEFAULT_STOP_TRIGGER_SERVICE = "/aima/middleware/stop"
DEFAULT_STATUS_TRIGGER_SERVICE = "/aima/middleware/status"
DEFAULT_INTERACTIVE_COMMAND_TOPIC = "/aima/middleware/command"

CONTROL_PERIOD_S = 0.02
HAL_STATE_PERIOD_S = 0.002

JOINT_NAME_TO_INDEX = {name: index for index, name in enumerate(JOINT_ORDER)}
