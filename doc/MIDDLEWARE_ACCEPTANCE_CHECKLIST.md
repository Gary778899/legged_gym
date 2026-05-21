# X2 Middleware Acceptance Checklist

This checklist is for validating the current `wbc_middleware` delivery inside the Ubuntu 22.04 devcontainer / Docker environment.

## 0. Scope

This checklist covers:

- devcontainer runtime readiness
- ROS2 + `aimdk_msgs` overlay readiness
- middleware node startup
- MuJoCo mock bridge startup
- HAL topic closed loop
- startup state machine
- command-source verification

This checklist does not cover:

- real robot deployment
- Docker image optimization

## 1. Preconditions

Run everything inside the devcontainer.

Expected environment:

- Ubuntu 22.04 container
- ROS2 Humble available
- Python deps available: `numpy`, `mujoco`, `onnxruntime`, `rclpy`
- repo opened at `/workspace/legged_gym`
- if MuJoCo viewer is used from Docker, the host display permission is available to the container

If the viewer is launched from a root container on Linux/X11, the host may need:

```bash
xhost +si:localuser:root
```

## 2. One-Time Overlay Setup

Open a terminal in the container and run:

```bash
source /opt/ros/humble/setup.bash
mkdir -p /workspace/.ros2_ws/src
ln -sfn /workspace/legged_gym/wbc_middleware/aimdk_msgs /workspace/.ros2_ws/src/aimdk_msgs
colcon build \
  --symlink-install \
  --base-paths /workspace/.ros2_ws/src \
  --build-base /workspace/.ros2_ws/build \
  --install-base /workspace/.ros2_ws/install
source /workspace/.ros2_ws/install/setup.bash
```

Verify imports:

```bash
python -c "import numpy, mujoco, onnxruntime; print('python deps ok')"
python -c "import rclpy; print('ros2 ok')"
python -c "from aimdk_msgs.msg import JointCommandArray; print('aimdk_msgs ok')"
```

Expected result:

- all three commands succeed

## 3. Static Sanity Check

From `/workspace/legged_gym`, run:

```bash
python -m py_compile \
  wbc_middleware/control_middleware.py \
  wbc_middleware/core/*.py \
  wbc_middleware/ros2/*.py \
  wbc_middleware/mock_bridge/*.py
```

Expected result:

- no syntax error output

## 4. Start Middleware Node

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
cd /workspace/legged_gym
python wbc_middleware/control_middleware.py
```

Check:

- node starts successfully
- initial state is `IDLE`
- no immediate import/runtime crash
- no automatic move-to-default action before trigger

## 5. Check Startup Service

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 service list | grep startup
```

Expected result:

- `/aima/middleware/startup` is present

Optional:

```bash
ros2 service type /aima/middleware/startup
```

Expected result:

- `std_srvs/srv/Trigger`

## 6. Start MuJoCo Mock Bridge

Terminal 3:

Headless mode:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
cd /workspace/legged_gym
python -m wbc_middleware.mock_bridge
```

Viewer mode:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
cd /workspace/legged_gym
python -m wbc_middleware.mock_bridge --viewer
```

Check:

- bridge starts successfully
- MuJoCo model loads successfully
- no immediate ROS2 / message / MuJoCo runtime crash
- with `--viewer`, the robot holds a nominal startup pose before the first external joint command arrives
- bridge log reports `hold_nominal_pose_until_first_command=True` when the default config is used

## 7. Check HAL Topics

Terminal 4:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 topic list | grep /aima
```

Expected topics:

- `/aima/hal/imu/torso/state`
- `/aima/hal/joint/leg/state`
- `/aima/hal/joint/leg/command`

Check state stream:

```bash
ros2 topic echo /aima/hal/joint/leg/state
```

Expected result:

- joint state messages are continuously published

## 8. Trigger Startup Sequence

Terminal 5:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 service call /aima/middleware/startup std_srvs/srv/Trigger "{}"
```

Expected middleware behavior in Terminal 1:

- transition `IDLE -> MOVE_TO_DEFAULT`
- transition `MOVE_TO_DEFAULT -> POLICY_HOLD`
- transition `POLICY_HOLD -> POLICY_ACTIVE`

Expected service result:

- `success: true`

## 9. Check Command Publication

Terminal 6:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 topic echo /aima/hal/joint/leg/command
```

Expected result:

- command messages appear after startup trigger
- command message contains 12 joints

## 10. Verify Closed Loop

Observe both:

- middleware logs
- MuJoCo bridge / viewer behavior

Expected result:

- MuJoCo responds only through `/aima/hal/joint/leg/command`
- no direct policy-to-MuJoCo shortcut is involved

## 11. Verify Trigger Rejection Outside IDLE

Call the startup service again after activation:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 service call /aima/middleware/startup std_srvs/srv/Trigger "{}"
```

Expected result:

- `success: false`
- response message explains current state is not `IDLE`

## 12. Verify Zero Command Source

Current default config:

- `wbc_middleware/config/x2_middleware.yaml`
- `command_source.type: zero`

Expected result in `POLICY_ACTIVE`:

- middleware runs with zero velocity command input

## 14. Verify Stop Service

Terminal 7:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 service call /aima/middleware/stop std_srvs/srv/Trigger "{}"
```

Expected result:

- `success: true` while middleware is in `MOVE_TO_DEFAULT`, `POLICY_HOLD`, or `POLICY_ACTIVE`
- middleware transitions into `STOPPING`
- middleware then transitions into `SAFE_HOLD`
- joint command topic continues publishing safe-hold commands according to `safety.safe_hold_mode`

## 15. Verify State Timeout Handling

With middleware and bridge running, stop the bridge process or otherwise interrupt HAL state publication.

Expected result in middleware logs:

- stale-state warning is reported
- middleware requests a safe stop
- middleware publishes safe-hold commands according to `safety.safe_hold_mode` instead of continuing policy execution

## 16. Verify Shutdown Safe Hold

While middleware is active, stop Terminal 1 with `Ctrl+C`.

Expected result:

- middleware publishes a final safe-hold command before exit when `safety.publish_safe_hold_on_shutdown: true`
- no uncaught exception is printed during shutdown

## 13. Verify Scripted Command Source

Edit `wbc_middleware/config/x2_middleware.yaml`:

```yaml
command_source:
  type: scripted
  segments:
    - duration_s: 3.0
      command: [0.3, 0.0, 0.0]
    - duration_s: 3.0
      command: [0.0, 0.0, 0.3]
```

Restart middleware and run the same checklist again from section 4.

Expected result:

- first segment drives forward command
- second segment drives yaw command
- scripted timing starts when middleware enters `POLICY_ACTIVE`, not when the node process starts
- for `commands_scale: [2.0, 2.0, 0.25]`, the observation command slice should show approximately:
  - first segment `[0.3, 0.0, 0.0] -> [0.6, 0.0, 0.0]`
  - second segment `[0.0, 0.0, 0.3] -> [0.0, 0.0, 0.075]`
- after the final segment, the last command remains active

## 14. Verify Interactive Command Topic

With middleware and bridge both running, publish:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
ros2 topic pub -r 10 /aima/middleware/command geometry_msgs/msg/Twist \
"{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}"
```

Expected result:

- interactive command overrides scripted / zero source
- middleware uses `(vx, vy, yaw) = (0.3, 0.0, 0.2)`
- for `commands_scale: [2.0, 2.0, 0.25]`, the observation command slice should show approximately `[0.6, 0.0, 0.05]` while publication is active

After stopping publication and waiting for timeout:

- middleware falls back to scripted source if configured
- otherwise falls back to zero source

## 15. Failure Notes

If `aimdk_msgs` import fails:

- re-run:

```bash
source /opt/ros/humble/setup.bash
source /workspace/.ros2_ws/install/setup.bash
```

If middleware import fails for ROS messages:

- confirm `aimdk_msgs` overlay has been built

If `python wbc_middleware/control_middleware.py` fails on module path:

- ensure current directory is `/workspace/legged_gym`

If MuJoCo bridge fails:

- confirm the XML path in `wbc_middleware/config/x2_middleware.yaml`
- confirm MuJoCo Python package imports successfully

If MuJoCo viewer fails to open from Docker:

- confirm the host display is exported into the container
- on Linux/X11 root containers, run `xhost +si:localuser:root` on the host before restarting the container

If the robot falls before startup trigger when using the viewer:

- confirm `mock_bridge.hold_nominal_pose_until_first_command: true` is enabled in `wbc_middleware/config/x2_middleware.yaml`

If scripted command starts on the wrong segment immediately after activation:

- confirm the middleware includes the fix that resets scripted command timing on transition to `POLICY_ACTIVE`

## 16. Pass Criteria

This delivery is accepted when all of the following are true:

- middleware starts in `IDLE`
- startup service exists and works
- mock bridge publishes HAL state topics
- middleware publishes HAL command topic
- startup state machine transitions correctly
- startup service is rejected outside `IDLE`
- MuJoCo responds through the HAL topic loop
- `zero`, `scripted`, and interactive command paths behave as expected

## 17. Acceptance Notes

The current delivery includes the following validated behaviors that are important for stable MuJoCo acceptance:

- `mock_bridge` can hold the robot at a nominal startup pose until the first `JointCommandArray` arrives
- `ScriptedCommandSource` timing is reset on transition to `POLICY_ACTIVE`, so scripted segments are aligned with policy takeover instead of process startup time
- stale HAL state now triggers a safe stop path instead of silently leaving the control loop idle
- shutdown safe-hold is published without the earlier `rclpy` context/shutdown exception path
- safe-hold behavior is now configurable through `safety.safe_hold_mode`, with `hold_position` preserved as the default acceptance mode

Known limitation:

- current acceptance covers the lower-body mock path only; upper-body stabilization commands are still an open item before real hardware deployment.
