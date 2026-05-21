# Full-Body Training Evaluation And History

## Goal

Evaluate whether the proposed "upper-body implicit PD + lower-body RL" architecture is a good next step after the deployment-side stabilization experiments failed to achieve stable standing in MuJoCo mock.

## Short Conclusion

Gemini's suggestion is feasible and is the right next direction.

However, it is **not** a pure deployment-side change. In this codebase, it is a **training-side architecture change**:

- switch the training asset from the current 12DoF lower-body-only model to a full-body model with active upper-body joints
- keep the policy action space at 12 lower-body actions
- inject upper-body PD targets inside the simulator torque path
- keep or strengthen orientation/contact-related rewards

The recommended workflow is:

1. Keep the current middleware/mock work as-is.
2. Start the training-side full-body work from the **current codebase**, not from an older commit.
3. Do that work on a **new branch or separate git worktree**, so middleware/mock validation and training iteration can proceed independently.

## Why Deployment-Side Compensation Was Not Enough

We completed several rounds of upper-body stabilization work in `wbc_middleware` and MuJoCo mock:

- split real HAL publishing by `leg / waist / arm / head`
- added upper-body fixed-hold command generation
- extended the MuJoCo mock model to include upper-body joints and actuators
- added a lightweight torso-based waist stabilization loop
- added a fall logger for quantitative diagnosis

### What the fall logs showed

The exported CSV logs consistently showed:

- the robot stays roughly fine through startup, then falls after entering `POLICY_ACTIVE`
- `torso_pitch` drifts continuously backward to about `-1.26 ~ -1.28 rad`
- `waist_pitch_target` saturates at its positive limit for long periods
- lower-body `action_norm` rises rapidly to around `8 ~ 11+`
- the "feet keep stepping after fall" behavior is a downstream symptom of the policy leaving its training distribution

### Interpretation

This strongly suggests the core problem is not just middleware gain tuning.

Once the upper body is turned into a dynamically free structure, the lower-body RL policy is no longer controlling the same system it was trained on. The base inertia, COM, and coupling changed enough that deployment-side compensation alone was not sufficient to recover stable standing.

This is valuable engineering evidence: it justifies the middleware/mock validation path and reduces risk before hardware deployment.

## Evaluation Of Gemini's Proposal

## 1. Is it feasible?

Yes, with caveats.

The proposal is technically sound:

- upper body remains dynamically active in simulation
- policy still outputs only lower-body actions
- upper-body joints are held near default posture by high-gain PD
- reward emphasizes stable torso orientation and clean foot-ground interaction

This is a common and practical stepping stone when the target policy should focus on locomotion first while the upper body is not yet policy-controlled.

## 2. Why it matches our current findings

Our recent experiments already showed that:

- fully free upper-body dynamics break the assumptions of the current lower-body-only policy
- deployment-side compensation is too weak and too late in the loop
- mock testing is valuable because it exposes this mismatch safely

Gemini's suggestion addresses the mismatch at the correct layer:

- not after policy inference
- not only in middleware
- but inside the training environment itself

## 3. What must change in this repository

The current training stack is still built around the 12DoF asset:

- `legged_gym/envs/x2/x2_config.py`
  - `asset.file` points to `resources/robots/x2/urdf/x2_12dof.urdf`
  - `num_actions = 12`
  - `default_joint_angles` currently define only lower-body joints
- `legged_gym/envs/base/legged_robot.py`
  - `_compute_torques()` assumes the control tensor aligns with the environment DOF control path
  - `self.torques`, `self.p_gains`, `self.d_gains`, and related buffers are part of that same torque pipeline
- `legged_gym/envs/x2/x2_env.py`
  - current observation layout is tightly tied to the lower-body setup and current observation dimension

There is already a full-body URDF available in the repository:

- `resources/robots/x2/urdf/x2_ultra.urdf`

That means we do **not** need to invent a full-body robot description from scratch for training.

## 4. Important caveat: asset swap alone is not enough

Gemini's idea is feasible, but with this codebase we cannot simply replace:

- `x2_12dof.urdf` -> `x2_ultra.urdf`

and leave everything else unchanged.

At minimum, we also need training-side code changes for:

### Torque routing

The environment should:

- keep policy action output at 12 lower-body commands
- internally construct a full-DOF target vector
- fill leg targets from the policy actions
- fill upper-body targets from fixed/default posture references
- apply high-gain PD on upper-body joints

### Observation definition

Current observation code is still shaped around the present lower-body setup.

We need to make an explicit choice:

1. keep observations mostly lower-body-centric and only retain base state needed for locomotion
2. or enlarge observations to include upper-body DOF state and retrain with a new observation definition

For the first stable walking demo, option 1 is the safer engineering choice.

### Default joint angles and gains

The training config must be expanded so all active upper-body joints have:

- default angles
- stiffness
- damping

Recommended initial default posture:

- `left_elbow_joint = -0.5`
- `right_elbow_joint = -0.5`
- all other upper-body joints near `0.0`

### Reward tuning

Gemini's reward suggestion is also feasible.

In fact, this repository already contains relevant reward terms such as:

- orientation penalty
- angular velocity penalty
- `contact_no_vel`

So the likely next step is **re-weighting**, not designing reward logic from zero.

Recommended direction:

- increase penalty on base roll/pitch deviation
- increase penalty on base angular velocity in roll/pitch
- strengthen foot-contact velocity penalty if it improves stance cleanliness
- avoid making reward changes too aggressive before the full-body PD hold is working

## Recommendation On Version Strategy

## Do not restart from an older git version

That is not recommended.

Reasons:

- the current middleware and mock work already produced valuable evidence
- the full-body mock model, fall logger, and split upper-body command path are useful infrastructure
- the current branch contains the latest understanding of the real HAL integration path
- going back would discard useful debug leverage and force duplicated work

## Recommended strategy

Use the **current codebase as the base**, but isolate the training work.

Recommended options:

### Option A: new branch from current HEAD

Good default choice if one person is iterating.

Example intent:

- keep current branch for middleware/mock integration history
- create a new branch for full-body training changes

### Option B: separate git worktree

Best choice if you want to keep middleware/mock experiments runnable in parallel while training code evolves independently.

This is especially useful because:

- the training-side changes will likely touch `legged_gym/envs/x2/*`
- the deployment-side changes already touch `wbc_middleware/*` and robot resources
- we may want to compare full-body training against the current deployment-only path without constantly stashing changes

## Recommended choice for this project

Prefer:

- **current HEAD + new branch or new worktree**

Do **not** start from a previous commit unless you later discover that the training branch specifically needs a cleaner asset history for reproducibility.

## Suggested Next Training Plan

1. Create a dedicated training branch or worktree.
2. Add a new X2 full-body training config instead of overwriting the existing 12DoF config.
3. Point that config to `resources/robots/x2/urdf/x2_ultra.urdf` or another confirmed full-body asset.
4. Keep `num_actions = 12`.
5. Modify the torque path so:
   - policy controls only leg joints
   - upper-body joints receive fixed/reference targets with high-gain PD
6. Keep the initial observation space as locomotion-focused, not full upper-body state heavy.
7. Re-weight orientation and contact penalties.
8. Train a first standing/walking policy.
9. Validate in MuJoCo with the existing middleware/mock infrastructure.

## Key Conversation Record

### Real HAL alignment

- Upper-body command publishing must follow real HAL split topics:
  - `leg`
  - `waist`
  - `arm`
  - `head`
- The old "unified command" assumption in the design document was incorrect.

### Upper-body state scope

- Upper-body state feedback is deferred.
- Current phase focuses on command generation, mock integration, and safe validation.

### Default upper-body posture

- `left_elbow_joint = -0.5`
- `right_elbow_joint = -0.5`
- all other upper-body joints default to `0.0`

### Mock/full-body validation outcome

- Upper-body commands were successfully published and consumed by the MuJoCo mock after the model and bridge were extended.
- Viewer tests showed that simply enabling upper-body dynamics made the robot unstable.
- Several deployment-side gain/stabilization attempts were tried.
- Quantitative fall logs showed that the policy leaves its training distribution shortly after entering active control.

### Engineering takeaway

- middleware + mock development was necessary and useful
- deployment-only upper-body stabilization is likely insufficient for stable standing/walking with the current lower-body-trained policy
- training-side adaptation is the right next step

## Final Recommendation

Proceed with a new **training branch/worktree** from the **current codebase**, and implement a new full-body training environment that keeps:

- lower-body RL actions
- upper-body implicit high-gain PD hold

This is the most direct path to a first stable walking demo while preserving the middleware/mock work that already reduced deployment risk.
