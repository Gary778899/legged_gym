# Full-Body Implicit PD Training Minimal Design

## Goal

Train the first stable X2 walking policy with:

- lower body controlled by RL
- upper body dynamically active in simulation
- upper body held by implicit high-gain PD toward a fixed reference posture

This design is intentionally minimal. It is meant to be easy for a basic coding agent to implement without changing the overall training framework.

## Non-Goals

This phase does **not** try to:

- make upper-body joints policy-controlled
- redesign the PPO pipeline
- add upper-body observations unless strictly necessary
- change middleware deployment logic
- optimize final reward quality beyond first stable standing/walking

## Core Idea

Keep the policy interface unchanged:

- policy action dimension stays `12`
- actions still correspond only to the leg joints

Change only the simulator-side actuation logic:

- load a full-body X2 asset with active upper-body joints
- build a full-DOF target internally
- fill leg targets from the policy action
- fill upper-body targets from a fixed default posture
- apply high-gain PD to the upper body inside the training environment

## Existing Constraints In This Repo

Current lower-body training setup:

- `legged_gym/envs/x2/x2_config.py`
  - uses `resources/robots/x2/urdf/x2_12dof.urdf`
  - `num_actions = 12`
- `legged_gym/envs/x2/x2_env.py`
  - observation shape and noise layout assume current lower-body setup
- `legged_gym/envs/base/legged_robot.py`
  - `_compute_torques()` currently assumes the control path is aligned with the action vector

Available full-body asset:

- `resources/robots/x2/urdf/x2_ultra.urdf`

## Recommended File Strategy

Do **not** overwrite the current X2 training setup.

Create a new full-body training variant in parallel.

Recommended new files:

- `legged_gym/envs/x2/x2_fullbody_env.py`
- `legged_gym/envs/x2/x2_fullbody_config.py`

Allowed alternative:

- keep `x2_env.py` and `x2_config.py` mostly untouched
- subclass `X2Robot` for the new full-body variant

## Required Joint Policy Split

### RL-controlled joints

The 12 lower-body joints remain policy-controlled:

- left hip pitch/roll/yaw
- left knee
- left ankle pitch/roll
- right hip pitch/roll/yaw
- right knee
- right ankle pitch/roll

### Implicit-PD-controlled joints

The upper body should be active in physics but not controlled by RL:

- waist: 3 joints
- arms: 14 joints
- head: 2 joints

## Default Upper-Body Reference Posture

Use this initial reference posture:

- `left_elbow_joint = -0.5`
- `right_elbow_joint = -0.5`
- all other upper-body joints = `0.0`

This matches the current middleware-side alignment and should be kept consistent across training and deployment.

## Minimal Implementation Plan

## 1. Create a new training config

Create `x2_fullbody_config.py`.

Base it on the current X2 config, but change:

- asset path:
  - from `x2_12dof.urdf`
  - to `x2_ultra.urdf`
- keep `num_actions = 12`
- expand `init_state.default_joint_angles` so **all active DOFs** in the full-body asset have defaults
- add stiffness/damping entries for upper-body joints

Recommended first-pass upper-body gains:

- waist: medium-high stiffness, moderate damping
- arm/head: high stiffness, moderate damping

Do not overtune yet. First target is stable training behavior, not perfect pose tracking.

## 2. Create a new environment class

Create `x2_fullbody_env.py`.

It should inherit from the current X2 environment or from `LeggedRobot`.

The new environment must preserve:

- 12D policy action interface
- current locomotion command interface
- current phase/gait logic if already working

## 3. Override the torque construction path

This is the most important change.

Implement a full-body torque builder that:

1. receives `actions` of shape `(num_envs, 12)`
2. creates a full target vector of shape `(num_envs, num_dof)`
3. maps the 12 action outputs only onto the lower-body joints
4. keeps upper-body joints at their default reference posture
5. computes PD torques for **all** active DOFs
6. clips torques by the asset torque limits

### Important rule

Do **not** enlarge the PPO action dimension.

The full-body logic must happen inside the environment torque path, not in the policy head.

## 4. Keep observations minimal

For the first version, do **not** add full upper-body joint observations unless needed.

Recommended observation policy:

- keep the current locomotion-centric observation layout
- keep lower-body DOF position/velocity observations
- keep base angular velocity and projected gravity
- keep phase and previous action terms

If the current code accidentally includes all DOFs after the asset swap, explicitly slice observation terms back to the 12 lower-body joints.

### Reason

The main objective is to make the plant more realistic while preserving a familiar policy interface.

## 5. Reward tuning

Do not rewrite the reward system from scratch.

Start from the current X2 reward structure and adjust weights only.

Priority reward adjustments:

- increase penalty on base roll/pitch orientation error
- increase penalty on base roll/pitch angular velocity
- strengthen foot-contact velocity penalty if needed

Keep the following idea in mind:

- first get a stable standing/walking policy
- then refine gait quality later

## 6. Keep deployment and training consistent

The upper-body reference posture used in training should match deployment defaults:

- same joint names
- same elbow default angles
- same semantic expectation that upper body is held, not policy-driven

## Minimal Acceptance Criteria

The implementation is acceptable when all of the following are true:

1. The new training env loads a full-body asset successfully.
2. PPO still sees a 12D action space.
3. Upper-body joints are active in simulation and receive nonzero PD torques.
4. Training runs without dimension mismatches.
5. The trained policy stands or walks more stably than the current lower-body-only-trained policy when evaluated with active upper-body dynamics.

## File-Level Task List For A Basic Agent

## Task A: Add config

Create:

- `legged_gym/envs/x2/x2_fullbody_config.py`

Must include:

- full-body asset path
- all default joint angles
- upper-body stiffness/damping
- unchanged `num_actions = 12`

## Task B: Add env

Create:

- `legged_gym/envs/x2/x2_fullbody_env.py`

Must include:

- lower-body action mapping
- full-body PD target assembly
- torque computation for all active DOFs

## Task C: Register the new env

Update whatever registry or import path is needed so training can launch the new full-body variant without replacing the old one.

## Task D: Validate dimensions

Check carefully:

- action tensor shape
- observation tensor shape
- default DOF vector shape
- torque tensor shape
- stiffness/damping vector shape

## Task E: Smoke test

Before long training:

- run a short launch
- confirm the asset loads
- confirm no shape mismatch
- confirm upper-body joints are active and held by PD

## Risks To Watch

### Risk 1: Observation dimension silently changes

If the environment starts reading all DOFs after switching to a full-body asset, the observation layout may silently stop matching the configured observation dimension.

Mitigation:

- explicitly slice lower-body indices where necessary
- verify `num_observations` against the actual concatenated tensor shape

### Risk 2: Torque buffer shape mismatch

The base class currently has places where action count and torque buffers are closely related.

Mitigation:

- verify that the torque tensor sent to Isaac Gym matches `num_dof`
- do not assume `num_actions == num_dof`

### Risk 3: Full-body gains too high

If upper-body PD gains are too aggressive, training may become noisy or unstable.

Mitigation:

- start high enough to hold posture
- but avoid extreme gains in the first pass

### Risk 4: Asset naming mismatch

If full-body URDF joint names do not match config dictionaries exactly, some gains may silently become zero.

Mitigation:

- verify joint names directly from the asset
- confirm every active upper-body joint has a default angle and matching gain rule

## Recommended First Delivery

The first development delivery should include only:

- new full-body config
- new full-body env
- torque path override
- reward weight adjustments
- smoke-test instructions

Do not bundle middleware or deployment edits into this training branch task.

## Final Instruction For The Implementing Agent

When in doubt, preserve these invariants:

- keep the old 12DoF training path working
- keep PPO action dimension at 12
- move full-body complexity into the environment, not into the policy API
- prefer the smallest change that produces a trainable full-body locomotion setup
