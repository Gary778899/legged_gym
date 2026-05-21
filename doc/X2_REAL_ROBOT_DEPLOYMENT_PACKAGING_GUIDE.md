# X2 Real Robot Deployment Packaging Guide

## Goal

This document describes a practical packaging workflow for deploying the current `x2_fullbody` middleware stack to a real robot environment.

The target is not just copying one script. The target package should include:

- exported ONNX policy
- ONNX metadata sidecar
- middleware config
- Python middleware source
- ROS2 message package
- any launch/start scripts needed on the robot side

## 1. What Needs To Be Packaged

For the current middleware path, the minimum deployment bundle should contain these parts.

### Policy artifacts

- `logs/x2_fullbody/<run_name>/model_<checkpoint>.onnx`
- `logs/x2_fullbody/<run_name>/onnx_metadata.json`

These two files are the deployment source of truth for:

- observation dimension
- action dimension
- recurrent state size
- action scale
- lower-body `kp/kd`
- normalization scales

### Middleware runtime code

- `wbc_middleware/control_middleware.py`
- `wbc_middleware/__init__.py`
- `wbc_middleware/core/*.py`
- `wbc_middleware/ros2/*.py`

If the real robot deployment does not use the mock bridge, `wbc_middleware/mock_bridge/*.py` is optional.

### Middleware config

- `wbc_middleware/config/x2_middleware.yaml`

This file must be edited for the real target before packaging.

### ROS2 message package

- `wbc_middleware/aimdk_msgs/`

This must be available inside the robot-side ROS2 workspace if the runtime still uses the same message definitions.

### Optional documentation

- `doc/CONTROL_MIDDLEWARE_ARCHITECTURE.md`
- this packaging guide

## 2. Files To Edit Before Packaging

Before making a deployment bundle, update `wbc_middleware/config/x2_middleware.yaml`.

The following fields should be checked carefully:

### Policy and metadata paths

- `policy_path`
- `metadata_path`

These should point to the selected exported ONNX run.

### Runtime topics

Check robot-side topic names:

- `imu_topic`
- `joint_state_topic`
- `joint_command_topic`
- `startup_trigger_service`
- `stop_trigger_service`
- `interactive_command_topic`

### Upper-body mode

If the real robot should match the currently validated standing semantics, keep the same logic:

- upper body enabled
- default-pose PD hold
- no extra stabilization offsets unless intentionally reintroduced

### Safety limits

Check:

- `safety.action_clip`
- `safety.position_delta_clip`
- `safety.safe_hold_mode`
- `safety.safe_hold_damping`

These should be chosen with the real actuator limits and startup behavior in mind.

## 3. Recommended Bundle Layout

Recommended deploy bundle:

```text
x2_deploy_bundle/
  policy/
    model_2000.onnx
    onnx_metadata.json
  config/
    x2_middleware.yaml
  wbc_middleware/
    __init__.py
    control_middleware.py
    core/
    ros2/
    aimdk_msgs/
  scripts/
    install.sh
    run_control_middleware.sh
    setup_ros_env.sh
  README.md
```

This keeps policy artifacts, config, and runtime code together and makes versioning easier.

## 4. Suggested Packaging Steps

### Step 1. Freeze the policy version

Choose one validated run and checkpoint:

- ONNX file
- matching `onnx_metadata.json`
- matching `x2_middleware.yaml`

Do not mix files from different runs.

### Step 2. Copy runtime sources into a clean bundle directory

Copy:

- `wbc_middleware/control_middleware.py`
- `wbc_middleware/core/`
- `wbc_middleware/ros2/`
- `wbc_middleware/aimdk_msgs/`
- selected ONNX and metadata
- edited YAML config

### Step 3. Prepare a ROS2 workspace on the target machine

Suggested layout:

```text
ros2_ws/
  src/
    aimdk_msgs/
    x2_deploy_bundle/
```

If your deployment environment already has a ROS2 workspace, place the package content under its `src/`.

### Step 4. Install Python dependencies

At minimum, the middleware runtime needs:

- `numpy`
- `onnxruntime`
- `rclpy`

If the real deploy path uses additional vendor SDK packages, install those too.

### Step 5. Build ROS2 interfaces

Typical ROS2 build step:

```bash
cd <ros2_ws>
colcon build --symlink-install
source install/setup.bash
```

### Step 6. Launch the middleware

Typical entry command:

```bash
python wbc_middleware/control_middleware.py
```

If the target machine requires a wrapper script, use one fixed launch script instead of ad hoc commands.

## 5. Recommended Helper Scripts

For repeatable deployment, prepare these scripts in the bundle.

### `setup_ros_env.sh`

Responsibilities:

- source ROS2
- source workspace `install/setup.bash`
- export any required environment variables

### `install.sh`

Responsibilities:

- create or update the ROS2 workspace
- copy package files
- install Python dependencies
- run `colcon build`

### `run_control_middleware.sh`

Responsibilities:

- source environment
- switch into the bundle or workspace directory
- run `python wbc_middleware/control_middleware.py`

## 6. Recommended Pre-Deploy Checks

Before running on hardware, verify:

- ONNX model path is correct
- metadata file matches the ONNX run
- middleware config matches the selected policy
- lower-body joint order matches the HAL joint state naming
- IMU topic semantics are correct for the real robot
- if the robot IMU is torso-mounted, the runtime must transform it to pelvis/root semantics before policy observation build

That last point is especially important, because the recent MuJoCo standing fix depended on restoring pelvis/root IMU semantics.

## 7. Suggested Acceptance Flow On Hardware

Recommended real-robot rollout order:

1. Bring up ROS2 topics and verify joint/IMU state freshness.
2. Start middleware in `IDLE`.
3. Trigger startup and verify `MOVE_TO_DEFAULT`.
4. Verify `POLICY_HOLD` with zero commands.
5. Verify zero-command standing before any walking command.
6. Only then test small command inputs.

## 8. Versioning Advice

Each deploy bundle should record:

- training experiment name
- run folder
- checkpoint number
- ONNX filename
- metadata filename
- middleware config revision
- deployment date

This makes rollback and comparison much easier when several robot-side tests are running in parallel.
