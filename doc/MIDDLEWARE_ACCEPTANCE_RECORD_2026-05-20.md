# X2 Middleware Acceptance Record (2026-05-20)

This document records the middleware acceptance session completed on May 20, 2026 in the Ubuntu 22.04 devcontainer environment.

## Environment

- Repository: `/workspace/legged_gym`
- Container OS: Ubuntu 22.04
- ROS2: Humble
- Runtime components exercised:
  - `wbc_middleware/control_middleware.py`
  - `python -m wbc_middleware.mock_bridge --viewer`
  - `/aima/middleware/startup`
  - `/aima/middleware/command`

## Host Display Note

For MuJoCo viewer from the root container, host X11 permission had to be enabled before restarting the compose stack:

```bash
xhost +si:localuser:root
```

## Acceptance Summary

Status: passed

Verified items:

- middleware starts in `IDLE`
- startup service `/aima/middleware/startup` is available
- mock bridge publishes HAL state topics
- middleware publishes HAL joint command topic
- startup state machine reaches policy execution path
- repeated startup trigger is rejected outside `IDLE`
- MuJoCo responds through the HAL topic loop
- `zero`, `scripted`, and interactive command paths behave as expected

## Observed Issues During Acceptance

### 1. Viewer startup could fail because of container display permissions

Resolution:

- enable host X11 access for the root container user
- restart the compose stack after the host-side `xhost` command

### 2. Robot could fall immediately when the viewer opened before middleware command publication

Observed effect:

- the robot fell before startup trigger
- middleware could remain in `MOVE_TO_DEFAULT` because pose error never recovered below the threshold

Resolution implemented:

- `mock_bridge.hold_nominal_pose_until_first_command: true`
- bridge now keeps a nominal startup pose until the first `JointCommandArray` arrives

### 3. Scripted command timing initially started too early

Observed effect:

- the first scripted segment could be skipped before `POLICY_ACTIVE`
- observation command slice jumped directly to the later yaw segment

Resolution implemented:

- `ScriptedCommandSource` timing is reset when middleware transitions to `POLICY_ACTIVE`

## Detailed Validation Results

### Startup service and state machine

Verified behavior:

- first call accepted startup: `IDLE -> MOVE_TO_DEFAULT`
- later call rejected startup outside `IDLE`

Example accepted response:

```text
std_srvs.srv.Trigger_Response(success=True, message='Startup sequence accepted: IDLE -> MOVE_TO_DEFAULT.')
```

Example rejected response:

```text
std_srvs.srv.Trigger_Response(success=False, message='Startup sequence already active; only IDLE can transition to MOVE_TO_DEFAULT (current=MOVE_TO_DEFAULT).')
```

### Zero command source

Config used:

```yaml
command_source:
  type: zero
```

Verified behavior:

- policy took over successfully
- robot remained stably standing in the viewer
- observation command slice stayed at `[0.0, 0.0, 0.0]` after scaling

### Scripted command source

Config used during final validation:

```yaml
command_source:
  type: scripted
  segments:
    - duration_s: 5.0
      command: [0.5, 0.0, 0.0]
    - duration_s: 3.0
      command: [0.0, 0.0, 0.3]
```

Verified behavior:

- first active segment appeared as `[1.0, 0.0, 0.0]` in the scaled observation command slice
- second active segment appeared as `[0.0, 0.0, 0.075]` in the scaled observation command slice
- segment order matched the expected `POLICY_ACTIVE` timing

### Interactive ROS topic override

Command used:

```bash
ros2 topic pub -r 10 /aima/middleware/command geometry_msgs/msg/Twist \
"{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}"
```

Verified behavior:

- interactive topic overrode the configured base source
- scaled observation command slice showed `[0.6, 0.0, 0.05]`
- robot behavior in the viewer reflected the override command

## Final Conclusion

The current `wbc_middleware` delivery passes the documented MuJoCo mock acceptance flow in the devcontainer environment, including viewer-based closed-loop verification and command-source validation.


## Phase 7 Hardening Addendum

Additional validation completed after the initial mock-acceptance session:

- `/aima/middleware/stop` service was validated
- stale HAL state timeout handling was validated
- shutdown safe-hold behavior was validated
- shutdown-path `rclpy` context errors were fixed and re-tested
- configurable `safe_hold_mode` support was added and validated in both `hold_position` and `damping` configurations

Final status for phase 7 hardening:

- passed in the devcontainer environment for the current lower-body mock path

Important remaining limitation:

- upper-body command stabilization is still missing for real hardware deployment, even though the lower-body MuJoCo model passes current validation
