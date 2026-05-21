# X2 Middleware Phase 7 Hardening Summary (2026-05-20)

This document summarizes the phase 7 development and validation results for `wbc_middleware` in the Ubuntu 22.04 devcontainer environment.

## 1. Scope

Phase 7 focused on pre-real deployment hardening rather than adding a new control feature.

Implemented scope:

- startup / stop / safe-hold state handling
- stale HAL state timeout handling
- shutdown-path cleanup
- action clipping and target-position safety clipping
- configurable safe-hold behavior
- Docker / devcontainer acceptance validation

Deferred scope:

- real robot deployment
- upper-body control integration
- best-effort tuning of damping-mode gains on hardware
- `MOVE_TO_DEFAULT` trajectory shaping for large initial posture errors

## 2. Implemented Changes

### 2.1 Middleware runtime hardening

Implemented in `wbc_middleware/ros2/control_node.py`:

- added `STOPPING` and `SAFE_HOLD` states
- added `/aima/middleware/stop` service
- added stale-state detection using `state_timeout_s`
- added safe stop request path for timeout and operator stop
- added shutdown safe-hold publication path
- fixed `Ctrl+C` shutdown ordering so middleware exits without `rclpy` context errors

### 2.2 Command safety guards

Implemented in `wbc_middleware/core/command_mapper.py` and metadata/config handling:

- action clipping before command mapping
- target position delta clipping relative to default pose
- optional per-joint lower / upper target clipping
- observation clipping support through runtime metadata

### 2.3 Configurable safe-hold mode

Implemented in `wbc_middleware/config/x2_middleware.yaml` and middleware command building:

- `safe_hold_mode: hold_position | damping`
- default remains `hold_position` to preserve current acceptance behavior
- optional `safe_hold_damping` vector can override the default damping-mode gain values

Current semantics:

- `hold_position`: publish default lower-body pose with normal PD gains
- `damping`: publish zero stiffness and nonzero damping using the current joint position as reference

## 3. Acceptance Result

Status:

- phase 7 acceptance passed in the devcontainer environment

Validated items:

- middleware startup in `IDLE`
- startup service presence and correct state-machine transitions
- mock bridge startup and closed-loop HAL topic operation
- command publication after startup trigger
- repeated startup trigger rejection outside `IDLE`
- `zero`, `scripted`, and interactive command-source behavior
- `/aima/middleware/stop` service behavior
- stale-state timeout handling
- shutdown safe-hold behavior without uncaught exceptions

Notes:

- a MuJoCo viewer segmentation fault was observed on viewer shutdown in the container environment
- this did not block middleware acceptance because the middleware-side stop / timeout / shutdown behavior completed correctly before viewer exit

## 4. Issues Found During Phase 7

### 4.1 Shutdown safe-hold initially failed

Observed behavior:

- `Ctrl+C` could produce `publisher's context is invalid`
- `rcl_shutdown already called` could be raised during shutdown

Cause:

- ROS shutdown could happen before middleware finished its own safe-hold publication path

Resolution:

- middleware now disables the default `rclpy` signal-handler shutdown path
- safe-hold is published before explicit `rclpy.shutdown(context=node.context)`
- shutdown logging and publication are guarded by context-validity checks

Status:

- resolved

### 4.2 Safe-hold semantics were too tied to MuJoCo convenience

Observed behavior:

- original safe-hold behavior always tried to hold the default pose
- this is acceptable for MuJoCo acceptance but not necessarily the right real-robot failure behavior

Resolution:

- added configurable `safe_hold_mode`
- current default remains `hold_position`
- `damping` mode is now available for hardware-oriented testing

Status:

- resolved structurally
- hardware gain tuning still pending

### 4.3 Upper body is not actively stabilized by middleware

Observed behavior:

- current policy and middleware command path only cover the 12 lower-body joints
- the `x2_12dof` MuJoCo model keeps the upper body effectively fixed as part of the simplified model structure
- on real hardware, leaving the upper body without explicit hold commands would be unsafe and would allow uncontrolled motion

Impact:

- current middleware package is acceptable for lower-body MuJoCo validation
- current command output is not yet sufficient for a real robot with a physically free upper body

Planned follow-up:

- add explicit upper-body command support to the deployment interface
- send fixed-position upper-body commands with sufficiently strong stabilization gains when lower-body policy is active or when safe-hold is entered

Status:

- open follow-up item before real deployment

### 4.4 `MOVE_TO_DEFAULT` is still a direct PD target, not a shaped takeover trajectory

Observed behavior:

- middleware currently sends the default pose target directly during `MOVE_TO_DEFAULT`
- no interpolation, rate limit, or staged gain scheduling is applied yet

Assessment:

- acceptable for the current operating assumption where the robot is suspended and already close to its nominal standing posture
- not yet ideal for large initial pose errors from a true zero-torque startup state

Planned follow-up:

- revisit only when the real startup procedure requires smoother takeover from larger posture error

Status:

- deferred, documented assumption

## 5. Recommended Next Steps

Before real-robot integration:

- add upper-body stabilization commands to the middleware output contract
- tune `safe_hold_damping` on hardware instead of reusing nominal `kd`
- verify the final hardware-side meaning of damping mode with the vendor motion-control board
- re-evaluate whether `MOVE_TO_DEFAULT` needs trajectory shaping after upper-body control is introduced

## 6. Current Deployment Readiness Statement

Current `wbc_middleware` status:

- suitable for lower-body HAL-style MuJoCo closed-loop validation in Docker/devcontainer
- suitable for pre-real deployment software integration work
- not yet sufficient for direct full-body real-robot deployment because upper-body stabilization commands are still missing
