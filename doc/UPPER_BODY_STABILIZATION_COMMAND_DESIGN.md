# X2 Upper-Body Stabilization Command Design

This document defines a small, implementation-ready design for adding upper-body stabilization commands to the current `wbc_middleware` stack.

## 1. Motivation

The current middleware only commands the 12 lower-body joints used by the policy.

This is acceptable for the current MuJoCo `x2_12dof` mock path because the simplified model keeps the upper body effectively fixed. It is not acceptable for real hardware deployment, because a physically free upper body would receive no explicit stabilization command and could swing or collapse when the middleware takes control or when safe-hold is entered.

The goal of this change is not to add upper-body policy control. The goal is to add a deployment-side fixed-pose stabilization layer for upper-body joints while preserving the existing lower-body policy path.

## 2. Design Goals

Required:

- keep the lower-body policy interface unchanged
- keep the middleware observation and action dimensions unchanged
- publish one unified HAL joint command message that contains both:
  - lower-body commands from the current policy / startup / safe-hold path
  - upper-body fixed-pose stabilization commands from config
- allow different gain profiles for:
  - startup / active control
  - safe-hold / failure handling
- keep the current MuJoCo `x2_12dof` acceptance path working

Not required in this phase:

- upper-body policy learning
- upper-body teleoperation
- full-body observation expansion
- MuJoCo full-body validation in the current 12-DoF mock model

## 3. Functional Requirements

The middleware shall support a configured set of upper-body joints and publish fixed-position PD commands for them.

The middleware shall support at least two upper-body stabilization modes:

- `active_hold`
  - used during `MOVE_TO_DEFAULT`, `POLICY_HOLD`, and `POLICY_ACTIVE`
  - intended to keep the upper body in a stable nominal posture while the lower body is controlled by the policy
- `safe_hold`
  - used during `STOPPING`, `SAFE_HOLD`, and shutdown safe-hold publication
  - intended to keep the upper body controlled during failures or middleware shutdown

The middleware shall support different gain sets for the two modes above.

The middleware shall support a configuration flag to disable upper-body command publication in lower-body-only mock environments.

## 4. Proposed Command Model

### 4.1 Joint groups

Split deployment commands into two groups:

- lower body
  - the existing 12 policy-controlled joints
- upper body
  - all joints that are not policy-controlled but must be stabilized on the real robot

### 4.2 Command composition

Before publishing `/aima/hal/joint/leg/command` or the real HAL-equivalent joint command topic, the middleware should build a combined joint command list:

1. build lower-body targets using the existing state machine and policy path
2. build upper-body fixed-pose targets using the current middleware state
3. merge both groups into one outgoing command message in the exact vendor-required joint order

### 4.3 Fixed-pose upper-body command semantics

For each configured upper-body joint:

- target position = configured nominal angle
- target velocity = `0`
- effort = `0`
- stiffness = mode-specific configured gain
- damping = mode-specific configured gain

## 5. Proposed Configuration Additions

Add a new top-level config section such as:

```yaml
upper_body:
  enabled: false
  publish_in_mock: false
  joint_names:
    - waist_yaw_joint
    - waist_pitch_joint
    - left_shoulder_pitch_joint
    - left_shoulder_roll_joint
    - left_shoulder_yaw_joint
    - left_elbow_joint
    - right_shoulder_pitch_joint
    - right_shoulder_roll_joint
    - right_shoulder_yaw_joint
    - right_elbow_joint
    - head_yaw_joint
    - head_pitch_joint
  default_angles: [ ... ]

  active_hold:
    kp: [ ... ]
    kd: [ ... ]

  safe_hold:
    mode: hold_position
    kp: [ ... ]
    kd: [ ... ]
```

Notes:

- `enabled: false` should remain the default until the real deployment path is ready.
- `publish_in_mock: false` allows the current `x2_12dof` mock path to keep running without requiring upper-body joints that do not exist in the simplified model.
- `safe_hold.mode` may later be extended, but this phase can begin with `hold_position` only for upper body.

## 6. Proposed Code Structure

Recommended additions:

```text
wbc_middleware/
  core/
    upper_body_config.py
    upper_body_command_builder.py
```

### `core/upper_body_config.py`

Responsibilities:

- parse and validate upper-body config
- expose:
  - whether upper-body command publication is enabled
  - joint names
  - default angles
  - active-hold gains
  - safe-hold gains
  - whether mock publication is enabled

### `core/upper_body_command_builder.py`

Responsibilities:

- build upper-body fixed-pose commands for:
  - active-hold mode
  - safe-hold mode
- return the same command-target structure used by lower-body command mapping

## 7. Middleware Integration Plan

### 7.1 Control node integration

`ControlNode` should:

- load upper-body config at startup
- decide whether upper-body commands are currently publishable in this runtime
- select upper-body command mode from middleware state:
  - `MOVE_TO_DEFAULT` -> `active_hold`
  - `POLICY_HOLD` -> `active_hold`
  - `POLICY_ACTIVE` -> `active_hold`
  - `STOPPING` -> `safe_hold`
  - `SAFE_HOLD` -> `safe_hold`
  - shutdown safe-hold -> `safe_hold`
- merge upper-body and lower-body commands before publication

### 7.2 State-machine mapping

Recommended behavior:

- `IDLE`
  - optional: publish nothing, same as current behavior
- `MOVE_TO_DEFAULT`
  - lower body: existing default-pose command behavior
  - upper body: active-hold fixed pose
- `POLICY_HOLD`
  - lower body: existing default-pose hold
  - upper body: active-hold fixed pose
- `POLICY_ACTIVE`
  - lower body: policy output
  - upper body: active-hold fixed pose
- `STOPPING`
  - lower body: existing safe-hold behavior
  - upper body: safe-hold fixed pose
- `SAFE_HOLD`
  - lower body: existing safe-hold behavior
  - upper body: safe-hold fixed pose

## 8. Mock Compatibility Strategy

The current `x2_12dof` MuJoCo model does not expose full upper-body joints.

Therefore the implementation shall not assume that upper-body joints exist in the current mock path.

Recommended rule:

- if `upper_body.enabled: false`, the current behavior remains unchanged
- if `upper_body.enabled: true` and `publish_in_mock: false`, upper-body commands are skipped in the current mock environment
- only after a full-body mock or real HAL target exists should upper-body commands be required in validation

This avoids breaking the current accepted lower-body mock workflow.

## 9. Validation Plan

### Stage 1: Config and unit-level validation

Checks:

- invalid upper-body config is rejected cleanly
- joint counts, angle counts, and gain counts must match
- `upper_body_command_builder` returns the expected number of commands
- command mode switches correctly between `active_hold` and `safe_hold`

### Stage 2: Middleware publication validation

Checks:

- when upper-body publication is disabled, current lower-body behavior is unchanged
- when enabled, outgoing command messages include both lower-body and upper-body joints in the expected order
- lower-body policy action mapping remains unchanged

### Stage 3: Hardware-oriented integration validation

Checks:

- upper body receives fixed-position stabilization commands while lower-body policy is active
- upper body remains controlled during safe stop and middleware shutdown
- no uncontrolled upper-body swing occurs when the lower-body policy stops

## 10. Risks and Tradeoffs

### 10.1 Joint-order mismatch risk

The final outgoing command order must match the vendor HAL expectation exactly.

Mitigation:

- define one canonical full deployment joint order for the real robot path
- validate command order against vendor SDK / message contract before hardware testing

### 10.2 Gain tuning risk

Overly large upper-body gains can create hard impacts or coupling with lower-body balance.

Mitigation:

- start from moderate stabilization gains
- tune on suspended hardware first
- keep `active_hold` and `safe_hold` gains separate

### 10.3 Mock-path breakage risk

Requiring upper-body commands too early could break the current accepted 12-DoF mock path.

Mitigation:

- keep upper-body publishing disabled by default
- guard mock publication with config

## 11. Recommended Implementation Sequence

1. Add upper-body config parsing and validation.
2. Add upper-body fixed-pose command builder.
3. Add command merge logic in `ControlNode`.
4. Keep `upper_body.enabled: false` by default.
5. Verify that the current lower-body mock acceptance still passes unchanged.
6. Add a hardware-facing config variant with `upper_body.enabled: true`.
7. Validate the combined output against the real joint-command interface.

## 12. Suggested Acceptance Criteria For The Follow-Up Task

The follow-up implementation is complete when all of the following are true:

- lower-body policy behavior is unchanged when upper-body publishing is disabled
- middleware can construct upper-body fixed-pose commands from config
- middleware publishes upper-body stabilization commands in `POLICY_ACTIVE`
- middleware preserves upper-body stabilization during `STOPPING`, `SAFE_HOLD`, and shutdown safe-hold
- current lower-body MuJoCo mock acceptance remains usable without requiring full-body model changes
- the design is ready for a later real-HAL / full-body validation step

## 13. Agent Handoff Notes

For the next implementation task, the agent should treat the following as hard constraints:

- do not change the 12-dim lower-body policy observation or action interface
- do not break the current `x2_12dof` mock acceptance path
- do not require upper-body joints to exist in the current mock model
- prefer additive changes in new helper modules over invasive rewrites of the policy path
- preserve current startup / stop / safe-hold semantics for the lower body
