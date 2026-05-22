# X2 Fullbody Standing Stability Summary

Date: 2026-05-21

Updated: 2026-05-22

## Scope

This note records the short conclusion for the recent `x2_fullbody` middleware + MuJoCo closed-loop standing validation.

Validation path:

`policy -> wbc_middleware -> mock_bridge -> MuJoCo -> mock_bridge -> wbc_middleware -> policy`

Reference runtime files:

- `wbc_middleware/config/x2_middleware.yaml`
- `wbc_middleware/ros2/control_node.py`
- `wbc_middleware/mock_bridge/mujoco_hal_bridge.py`

Reference log:

- `logs/csv/control_middleware_fall_log.csv`

## Final Result

The robot can now stand stably in the middleware-driven MuJoCo closed loop.

The latest CSV confirms that the previous fast-divergence failure mode is gone:

- total log duration: about `34.18 s`
- `POLICY_ACTIVE` duration: about `33.62 s`
- `torso_pitch` range: about `[-0.071, 0.026] rad`
- `torso_roll` range: about `[-0.003, 0.022] rad`
- `torso_pitch` never crosses `0.1 rad`
- `waist_pitch_target` stays at `0.0`, which means the stable result is achieved without extra waist stabilization offsets

This is strong evidence that the standing behavior is truly stable, not just visually improved for a short interval.

## Key Fixes That Mattered

### 1. Middleware phase was corrected to match training

The middleware gait phase is now aligned with the training observation semantics:

- training expects normalized phase in `[0, 1)`
- observation uses `sin(2*pi*phase)` and `cos(2*pi*phase)`

This removed a major observation mismatch.

### 2. Upper-body deployment semantics were aligned with training

For `x2_fullbody` training:

- upper body is not part of the policy action space
- upper body is not part of the lower-body observation slice
- but upper-body joints are still held by PD around the default pose

The middleware config was aligned to this behavior:

- upper body remains enabled in mock
- active waist stabilization offsets were disabled
- effective upper-body `kp/kd` scaling was reset to `1.0`

### 3. IMU/base semantics were corrected in the mock bridge

This was the most important final fix.

Training uses floating-base root state semantics for:

- base orientation
- base angular velocity
- projected gravity

The mock bridge previously mixed:

- torso orientation
- pelvis/root angular velocity

This was corrected in `wbc_middleware/mock_bridge/mujoco_hal_bridge.py` so that the IMU used by the policy now matches pelvis/root semantics consistently.

### 4. Previous-action observation was aligned with safety clipping

The middleware now feeds the policy the safe-applied action equivalent in the next observation, not the raw policy output.

Before this fix:

- the ONNX policy output was stored directly as `state.last_action`
- safety clipping could still reduce the target before command publication
- the next observation could therefore claim the previous action was larger than what the robot actually received

After this fix:

- action mapping applies `action_clip`, `position_delta_clip`, and optional joint position clipping
- the final safe target position is converted back to normalized action space
- that safe action equivalent becomes the next observation's previous-action field

This better matches training semantics, where the action term in the observation represents the action that actually drove the simulated controller path.

## Practical Interpretation

The current result means:

- the middleware closed loop is working
- the `x2_fullbody` ONNX policy is working inside the middleware path
- lower-body policy control plus fixed upper-body PD hold is now consistent enough to achieve stable standing in MuJoCo
- previous-action observation history is now consistent with middleware safety clipping

This also confirms that the previous instability was mainly caused by deployment-side semantic mismatches, not only by policy quality.

## Remaining Notes

The robot still shows a small standing pitch bias around `-0.06 rad`, but this does not prevent stable standing.

Possible later follow-up:

- reduce steady-state pitch bias
- validate command tracking beyond zero-command standing
- align runtime command semantics with training command semantics before serious turning tests
- repeat the same checks on real-HAL-connected deployment
