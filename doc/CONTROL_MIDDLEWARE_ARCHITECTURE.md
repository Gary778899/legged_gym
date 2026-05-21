# X2 Control Middleware Architecture

This document defines the proposed architecture for X2 control middleware development in this repository.

Current project priority:

- Keep `Train -> Sim2Sim` working for policy development.
- Build a deploy-oriented `control_middleware` that matches the vendor HAL interface.
- Validate `ONNX policy + control_middleware` in MuJoCo before real-robot deployment.

This design assumes:

- Robot target: X2 lower body only, 12 DoF
- Policy frequency: 50 Hz
- HAL state update frequency: 500 Hz
- MuJoCo mock test should reuse the same ROS2 HAL-style topics as future deployment
- Action semantic for mock test and deployment: PD target

## 1. Goals

The middleware stack should solve four problems:

1. Subscribe to HAL-like robot state.
2. Build observations exactly as training expects.
3. Run recurrent ONNX policy inference with managed LSTM state.
4. Publish HAL-style joint commands that can drive either:
   - a MuJoCo mock bridge, or
   - the real SDK/HAL pipeline later.

The MuJoCo mock test is not just a policy demo. It is a deployment-interface validation stage.

## 2. Design Principles

### 2.1 Reuse HAL topic contracts

The middleware should continue to consume and publish ROS2 topics compatible with the vendor HAL interface, for example:

- `/aima/hal/imu/torso/state`
- `/aima/hal/joint/leg/state`
- `/aima/hal/joint/leg/command`

Reason:

- This keeps the mock test close to real deployment.
- Topic naming, message typing, QoS, joint ordering, and control semantics are validated early.
- Existing rosbag replay workflows remain useful.

### 2.2 Keep middleware logic independent from transport

Although ROS2 topics are the external contract, the core logic should be separated from ROS2 node boilerplate.

This makes it easier to:

- unit-test observation logic offline
- compare middleware output against training config
- replace input/output adapters later without rewriting policy logic

### 2.3 Avoid modifying existing MuJoCo deployment scripts

The current `deploy/deploy_mujoco` flow should remain intact for existing sim2sim policy checks.

A separate bridge path should be added for middleware integration tests.

## 3. Proposed Repository Layout

Recommended target layout:

```text
wbc_middleware/
  aimdk_msgs/
  config/
    x2_middleware.yaml
  core/
    constants.py
    robot_state.py
    observation_builder.py
    onnx_policy_runner.py
    command_mapper.py
    metadata_loader.py
  ros2/
    control_node.py
    hal_state_buffer.py
    hal_command_publisher.py
  mock_bridge/
    mujoco_hal_bridge.py
    mujoco_command_adapter.py
  tools/
    replay_rosbag_notes.md
  control_middleware.py
  motocontrol.py
  policy.onnx
```

Notes:

- `control_middleware.py` can be retained temporarily as an entry script during migration.
- `motocontrol.py` remains reference code from the vendor side.
- `aimdk_msgs/` remains the SDK message package.

## 4. Main Components

## 4.1 Middleware core

These modules should contain the deploy-critical logic and avoid direct ROS2 dependencies.

### `core/constants.py`

Responsibilities:

- define canonical X2 lower-body joint order
- define default topic names if needed
- define default observation dimension and action dimension

Primary outputs:

- `JOINT_ORDER`
- `NUM_ACTIONS = 12`
- `NUM_OBS = 47`

### `core/robot_state.py`

Responsibilities:

- hold the latest robot state snapshot used by policy inference
- represent:
  - IMU quaternion
  - body angular velocity
  - 12 joint positions
  - 12 joint velocities
  - last action
  - timestamps if needed later

Primary inputs:

- ROS2 state callbacks

Primary outputs:

- current state snapshot for observation building

### `core/metadata_loader.py`

Responsibilities:

- load deployment metadata from:
  - ONNX embedded metadata when available
  - sidecar file when available
  - fallback YAML config when needed
- expose:
  - `num_observations`
  - `num_actions`
  - `rnn_hidden_size`
  - `rnn_num_layers`
  - `action_scale`
  - `kp/kd`
  - observation scales

Reason:

- avoids hardcoding LSTM hidden size and related deployment parameters
- reduces drift between training export and deployment runtime

### `core/observation_builder.py`

Responsibilities:

- construct the 47-dim observation vector exactly matching training

Expected observation order for current X2 policy:

1. base angular velocity, scaled: 3
2. projected gravity: 3
3. command input, scaled: 3
4. joint position offset from default: 12
5. joint velocity, scaled: 12
6. previous action: 12
7. gait phase sin/cos: 2

Primary inputs:

- `RobotState`
- command input vector `[vx, vy, yaw]`
- phase generator state
- metadata/config scales

Primary outputs:

- `obs: np.ndarray` with shape `(1, 47)`

Important rule:

- This module is a source-of-truth component and must stay aligned with
  [legged_gym/envs/x2/x2_env.py](/home/gary/Projects/legged_gym/legged_gym/envs/x2/x2_env.py)
  and
  [legged_gym/envs/base/legged_robot.py](/home/gary/Projects/legged_gym/legged_gym/envs/base/legged_robot.py).

### `core/onnx_policy_runner.py`

Responsibilities:

- load ONNX model
- inspect inputs/outputs
- allocate recurrent state from metadata or ONNX shape
- run inference
- manage memory reset

Primary inputs:

- observation vector `(1, 47)`
- hidden state
- cell state

Primary outputs:

- action `(12,)`
- updated hidden state
- updated cell state

Required behaviors:

- support recurrent policy inputs:
  - `obs`
  - `hidden_state`
  - `cell_state`
- expose `reset_memory()`

### `core/command_mapper.py`

Responsibilities:

- map raw policy action to HAL-compatible PD joint command

For current X2 lower-body policy:

- target position = `default_dof_pos + action * action_scale`
- target velocity = `0`
- effort = `0`
- stiffness = metadata/config kp
- damping = metadata/config kd

Primary inputs:

- raw policy action `(12,)`
- default joint pose
- action scale
- kp/kd configuration

Primary outputs:

- a structured joint command object or dict before ROS2 publishing

This module should also be the natural place for later additions such as:

- action clipping
- command smoothing
- rate limiting
- safety guards

## 4.2 ROS2 middleware adapter layer

These modules wrap the core logic into nodes that talk to the HAL-style ROS2 interface.

### `ros2/hal_state_buffer.py`

Responsibilities:

- subscribe to:
  - `/aima/hal/imu/torso/state`
  - `/aima/hal/joint/leg/state`
- keep the latest complete state snapshot
- handle high-rate state updates at 500 Hz

Primary outputs:

- latest valid `RobotState`

Design note:

- state callbacks should not run policy inference directly
- they should only update cached state

### `ros2/hal_command_publisher.py`

Responsibilities:

- publish `JointCommandArray` to:
  - `/aima/hal/joint/leg/command`

Primary inputs:

- mapped PD target command from `command_mapper`

Primary outputs:

- HAL-compatible ROS2 command topic

### `ros2/control_node.py`

Responsibilities:

- own the 50 Hz control loop
- read the latest state from `hal_state_buffer`
- update phase
- build obs
- run ONNX policy
- map action to PD target
- publish command

Primary inputs:

- latest HAL state snapshot
- command input source
- middleware config

Primary outputs:

- `/aima/hal/joint/leg/command`

Suggested internal loop:

1. read latest state snapshot
2. update phase for current control tick
3. build observation
4. infer action from ONNX policy
5. store action as `last_action`
6. build PD joint command
7. publish command

Control timing:

- middleware control loop: 50 Hz
- state subscription updates: up to 500 Hz
- each 50 Hz tick consumes the most recent state snapshot

## 4.3 MuJoCo mock bridge layer

This is a separate integration path and should not alter the existing `deploy/deploy_mujoco` scripts unless later reuse becomes clearly beneficial.

### `mock_bridge/mujoco_hal_bridge.py`

Responsibilities:

- run MuJoCo simulation or connect to a MuJoCo simulation driver
- publish HAL-style state topics derived from MuJoCo state
- subscribe to HAL-style joint command topic
- apply received PD target command to MuJoCo actuators
- optionally hold a nominal startup pose until the first external joint command arrives

Primary outputs:

- `/aima/hal/imu/torso/state`
- `/aima/hal/joint/leg/state`

Primary inputs:

- `/aima/hal/joint/leg/command`

This is the key adapter that makes MuJoCo look like a robot HAL endpoint.

### `mock_bridge/mujoco_command_adapter.py`

Responsibilities:

- convert `JointCommandArray` into MuJoCo control semantics
- for current design, implement PD target behavior:
  - target position
  - stiffness
  - damping
- compute actuator command or desired torque as required by MuJoCo model setup

Primary inputs:

- commanded joint target state
- current MuJoCo joint state

Primary outputs:

- control values applied into MuJoCo simulation

## 5. Runtime Data Flow

## 5.1 MuJoCo mock-test path

```text
MuJoCo sim
  -> mock_bridge optionally holds nominal startup pose until first command
  -> mock_bridge publishes /aima/hal/imu/torso/state
  -> mock_bridge publishes /aima/hal/joint/leg/state
  -> control_node caches state
  -> control_node builds obs
  -> control_node runs ONNX policy
  -> control_node publishes /aima/hal/joint/leg/command
  -> mock_bridge receives command
  -> mock_bridge applies PD target in MuJoCo
  -> robot moves in MuJoCo
```

This path validates:

- topic interface correctness
- observation correctness
- LSTM state handling
- action-to-command mapping
- PD command application behavior

## 5.2 Future real-robot path

```text
Vendor HAL / SDK
  -> /aima/hal/imu/torso/state
  -> /aima/hal/joint/leg/state
  -> control_node
  -> /aima/hal/joint/leg/command
  -> Vendor HAL / SDK
  -> robot
```

The key objective is for the `control_node` and core modules to remain unchanged between MuJoCo mock test and real deployment.

## 6. Command and State Interface

## 6.1 State topics

For current scope, middleware should consume:

- IMU:
  - topic: `/aima/hal/imu/torso/state`
  - type: `sensor_msgs/msg/Imu`
- Lower-body joint state:
  - topic: `/aima/hal/joint/leg/state`
  - type: `aimdk_msgs/msg/JointStateArray`

The middleware should reorder joint messages according to canonical joint order:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

## 6.2 Command topic

The middleware should publish:

- topic: `/aima/hal/joint/leg/command`
- type: `aimdk_msgs/msg/JointCommandArray`

Each `JointCommand` should contain:

- `name`
- `position`
- `velocity`
- `effort`
- `stiffness`
- `damping`

For current RL deployment:

- `position`: policy-mapped PD target
- `velocity`: `0.0`
- `effort`: `0.0`
- `stiffness`: config or metadata value
- `damping`: config or metadata value

## 7. Timing Model

## 7.1 State path

- MuJoCo bridge or real HAL publishes state at 500 Hz
- middleware caches the latest state

## 7.2 Control path

- middleware main loop runs at 50 Hz
- each tick uses latest available state snapshot

## 7.3 Phase evolution

For the current 50 Hz lower-body policy:

- control step `dt = 0.02 s`
- phase update should align with policy expectation

For now, phase management should be made explicit in middleware config rather than left as an inline magic number.

## 8. Configuration Strategy

Recommended runtime config file:

`wbc_middleware/config/x2_middleware.yaml`

Suggested contents:

- ONNX model path
- topic names
- control frequency
- state timeout threshold
- command source settings
- default command input
- gait period
- fallback kp/kd
- fallback observation scales

Priority order for configuration values:

1. ONNX metadata
2. deployment YAML
3. hardcoded fallback constants

This avoids accidental divergence from training/exported policy settings.

## 9. Command Input Strategy

Velocity command adjustment is part of the deployment workflow, but it should be introduced in stages.

The middleware should not hardcode one specific command source. Instead, it should consume commands through an abstract `CommandSource` interface.

Recommended command source variants:

- `ZeroCommandSource`
  - always outputs `[0.0, 0.0, 0.0]`
- `ScriptedCommandSource`
  - outputs preplanned command segments for deterministic mock tests
- `KeyboardCommandSource`
  - for local operator control during mock tests
- `GamepadCommandSource`
  - for operator-friendly interactive testing
- `RosTopicCommandSource`
  - subscribes to an external ROS2 command topic and forwards that command to middleware

For the current phase, only the first two are required:

- `ZeroCommandSource`
- `ScriptedCommandSource`

The currently implemented runtime also accepts an interactive ROS2 `Twist` topic override, which takes precedence over the configured base command source while messages remain fresh.

Reason:

- they are enough to validate the middleware closed loop
- they reduce debugging complexity in the first integration phase
- they make MuJoCo mock tests reproducible

Scripted command timing should be aligned with policy takeover. In the current validated implementation, scripted timing is reset when middleware enters `POLICY_ACTIVE`, so segment 0 starts when the policy begins consuming commands.

Recommended initial scripted command pattern:

```text
0s   - 2s   : [0.0, 0.0, 0.0]
2s   - 6s   : [0.3, 0.0, 0.0]
6s   - 10s  : [0.0, 0.0, 0.3]
10s+        : [0.0, 0.0, 0.0]
```

After the minimal mock-test pipeline is stable, `KeyboardCommandSource` and `GamepadCommandSource` should be added.

## 10. Startup and Safety State Machine

For real deployment, the robot can enter a zero-damping or otherwise unsafe passive state after the native motion-control board is disengaged. Because of that, policy inference must not start immediately.

The middleware should own a minimal startup state machine from the beginning, even if MuJoCo uses a simplified version of it.

Recommended states:

- `IDLE`
  - middleware started, but no control command sent yet
- `MOVE_TO_DEFAULT`
  - send PD targets to bring the robot to the default lower-body pose
- `POLICY_HOLD`
  - hold default pose and keep command input clamped to zero
- `POLICY_ACTIVE`
  - run full policy inference and accept external velocity command input
- `STOPPING`
  - transition back to a safe hold state
- `SAFE_HOLD`
  - keep robot at default or last safe pose with PD holding

Recommended startup sequence:

1. start middleware
2. initialize ONNX runtime and reset LSTM memory
3. enter `MOVE_TO_DEFAULT`
4. wait until joint positions are sufficiently close to default pose
5. hold the default pose for a short stabilization period
6. reset LSTM memory again before policy activation
7. enter `POLICY_HOLD`
8. keep command source at zero command
9. after operator confirmation or startup timeout, enter `POLICY_ACTIVE`
10. allow nonzero external velocity command input

Recommended shutdown sequence:

1. stop accepting nonzero command input
2. transition to `STOPPING`
3. return to zero command or default pose hold
4. enter `SAFE_HOLD`

For the mock-test phase, this state machine can be simplified but should still be present structurally.

For MuJoCo viewer acceptance, the validated bridge behavior is to keep the robot in a nominal pose before the first middleware joint command arrives. This avoids early falls that would otherwise block `MOVE_TO_DEFAULT` and produce false-negative acceptance failures.

At minimum, mock test should include:

- `MOVE_TO_DEFAULT`
- `POLICY_HOLD`
- `POLICY_ACTIVE`

## 11. Validation Plan

Validation should proceed in stages.

### Stage 1: Offline middleware correctness

Goal:

- verify observation dimension, command dimension, and recurrent state handling

Checks:

- obs shape is `(1, 47)`
- action shape is `(12,)`
- LSTM states reset and advance correctly
- joint reorder is correct

### Stage 2: Rosbag replay through HAL topics

Goal:

- verify state subscription and observation building against recorded real data

Checks:

- middleware subscribes correctly
- observation fields are numerically reasonable
- no missing-joint ordering issues

### Stage 3: MuJoCo mock HAL integration

Goal:

- verify full closed loop:
  `state -> obs -> ONNX -> command -> MuJoCo motion`

Checks:

- mock bridge publishes HAL topics correctly
- middleware emits valid `JointCommandArray`
- MuJoCo robot responds as expected

### Stage 4: Pre-deployment dry run

Goal:

- verify runtime behavior in the same Ubuntu 22.04 Docker environment intended to match the robot board

Checks:

- ROS2 dependencies available
- ONNX runtime available
- topic timing acceptable
- no path or packaging issues

## 12. Development Phases

The middleware project should be developed in explicit stages.

## Phase 0: Architecture confirmation

Goal:

- confirm component boundaries and data flow before refactoring

Deliverables:

- this architecture document

Status:

- completed after review and agreement

## Phase 1: Minimal middleware refactor

Goal:

- separate deploy-critical logic from the current monolithic prototype

Scope:

- extract `constants.py`
- extract `robot_state.py`
- extract `observation_builder.py`
- extract `onnx_policy_runner.py`
- keep ROS2 wiring simple

Deliverables:

- reusable middleware core modules
- no change yet to overall behavior

Success criteria:

- current `control_middleware.py` behavior is preserved
- observation output remains identical to the current prototype

## Phase 2: Command publishing and metadata cleanup

Goal:

- complete the output side of the middleware path

Scope:

- add `hal_command_publisher.py`
- map action to `JointCommandArray`
- remove hardcoded LSTM hidden size
- read deploy parameters from metadata or config

Deliverables:

- middleware publishes `/aima/hal/joint/leg/command`
- ONNX recurrent state is initialized from metadata/config

Success criteria:

- middleware produces valid ROS2 command messages at 50 Hz
- command values match expected PD target semantics

## Phase 3: Startup state machine

Goal:

- make policy startup compatible with future real-robot safety requirements

Scope:

- add `MOVE_TO_DEFAULT`
- add `POLICY_HOLD`
- add `POLICY_ACTIVE`
- add LSTM reset behavior tied to state transitions

Deliverables:

- middleware no longer starts directly in free-running policy mode

Success criteria:

- middleware can bring the robot or simulator model to default pose
- policy activation happens only after hold/confirmation logic

## Phase 4: MuJoCo mock bridge

Goal:

- validate the full deployment path through HAL-style ROS2 topics

Scope:

- create dedicated MuJoCo bridge node
- publish fake HAL state topics
- subscribe to middleware command topic
- apply PD targets inside MuJoCo

Deliverables:

- closed-loop MuJoCo mock test path

Success criteria:

- MuJoCo motion is driven entirely through the middleware command topic
- no direct policy-to-MuJoCo shortcut remains in this test path

## Phase 5: Deterministic command-source testing

Goal:

- validate policy behavior under controlled command inputs

Scope:

- add `ZeroCommandSource`
- add `ScriptedCommandSource`

Deliverables:

- repeatable forward/yaw scripted tests

Success criteria:

- scripted command segments produce stable and interpretable robot behavior

## Phase 6: Interactive command input

Goal:

- allow operator-driven testing

Scope:

- add keyboard teleop or gamepad teleop
- optionally add ROS2 command topic input

Deliverables:

- interactive velocity control during mock tests

Success criteria:

- operator can adjust velocity commands without modifying test scripts

## Phase 7: Pre-real deployment hardening

Goal:

- prepare the middleware stack for real-board deployment

Scope:

- Docker validation against Ubuntu 22.04 environment
- startup/shutdown behavior review
- state timeout handling
- command clipping and safety guards

Deliverables:

- deploy-ready middleware package candidate

Success criteria:

- middleware behavior is stable in the target software environment
- no unresolved interface ambiguity remains before hardware integration

### Phase 7 implementation status (2026-05-20)

Implemented and validated in Docker/devcontainer:

- shutdown path cleanup and explicit safe-hold publication
- stale HAL state timeout detection and safe-stop transition
- `/aima/middleware/stop` service
- action clipping and position safety clipping
- configurable `safe_hold_mode` with `hold_position` and `damping` variants

Current status:

- the lower-body mock pipeline passes phase 7 acceptance in the devcontainer environment
- the software stack is ready for continued pre-real integration work
- full real-robot readiness is still blocked by missing upper-body stabilization commands

Known follow-up items before hardware integration:

- add explicit upper-body fixed-pose stabilization commands with deployment-oriented gains
- tune `safe_hold_damping` on hardware instead of assuming nominal `kd` is sufficient
- revisit `MOVE_TO_DEFAULT` smoothing only if the real startup posture can differ significantly from nominal stand

## 13. Immediate Refactor Plan

Recommended short-term implementation sequence:

1. Keep the current `control_middleware.py` as the baseline reference.
2. Extract shared constants and observation logic into `core/`.
3. Replace hardcoded LSTM size with metadata-driven allocation.
4. Add ROS2 publisher for `/aima/hal/joint/leg/command`.
5. Add the minimal startup state machine with:
   - `MOVE_TO_DEFAULT`
   - `POLICY_HOLD`
   - `POLICY_ACTIVE`
6. Add `ZeroCommandSource` and `ScriptedCommandSource`.
7. Add a dedicated MuJoCo mock bridge node.
8. Verify the closed loop in Docker with 50 Hz control and 500 Hz state publication.

## 14. Non-Goals for This Phase

The following are intentionally out of scope for the current phase:

- upper-body control in the training/policy scope
  Note: deployment-side upper-body stabilization commands are still required before real hardware use.
- AMP / SONIC integration
- blending with WBC policies
- direct modification of the existing `deploy/deploy_mujoco` validation scripts
- broad refactor of training code

## 15. Summary

The recommended architecture is:

- keep the HAL-style ROS2 topic interface
- separate middleware core logic from ROS2 transport
- create a dedicated MuJoCo bridge node for mock test
- include a minimal startup/safety state machine early
- stage command input development instead of building all teleop options up front
- validate the exact future deployment path in simulation before touching real hardware

This gives the project a stable path from:

`trained ONNX policy`
-> `control_middleware`
-> `HAL-style command`
-> `MuJoCo mock test`
-> `future real robot deployment`
