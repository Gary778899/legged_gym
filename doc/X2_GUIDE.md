# X2 Training and Sim2Real Guide

This document is the consolidated X2-focused guide for this repository.

It merges the key content from:
- `doc/Notes.md`
- `doc/README.md`
- `doc/setup_en.md`

Source files are kept unchanged.

## 1. Project Scope

Primary scope in this branch:
- Train and improve locomotion policy for X2 in Isaac Gym.
- Validate policy in MuJoCo (sim2sim).
- Prepare for X2 sim2real workflow.

Current repository focus:
- X2 is the main target.
- Non-target robot content has been removed from active workflows.

## 2. Workflow Overview

Recommended pipeline:

`Train -> Play -> Sim2Sim -> Sim2Real`

- Train: optimize policy in Isaac Gym.
- Play: verify policy behavior and export TorchScript policy.
- Sim2Sim: run exported policy in MuJoCo to check transfer quality.
- Sim2Real: deploy to real robot once hardware interface/config is ready.

## 3. Setup and Installation

## 3.1 System Requirements

- OS: Ubuntu 18.04 or later
- GPU: NVIDIA GPU
- Driver: 525 or later recommended

## 3.2 Create Environment

```bash
uv create -n python=3.8
```

## 3.3 Install Core Dependencies

### PyTorch

```bash
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Isaac Gym

1. Download Isaac Gym from NVIDIA.
2. Install:

```bash
cd isaacgym/python
uv pip install -e .
```

3. Verify:

```bash
cd examples
python 1080_balls_of_solitude.py
```

### rsl_rl

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
uv pip install -e .
```

### This repository

```bash
cd legged_gym
uv pip install -e .
```

## 4. Command Examples (X2)

## 4.1 Training

```bash
python legged_gym/scripts/train.py --task x2 --headless
```

Example with explicit experiment settings:

```bash
python legged_gym/scripts/train.py --task x2 --headless --num_envs 1000 --max_iterations 3000 --experiment_name x2_reward_v2 --run_name seed1_debug
```

## 4.2 Play and Export

```bash
python legged_gym/scripts/play.py --task x2 --experiment_name x2_reward_v2 --load_run Mar11_14-22-10_seed1_debug --checkpoint 3000
```

Notes:
- `play.py` is used for policy validation and export.
- Exported policy is usually saved under `logs/<experiment_name>/exported/policies/`.

## 4.3 MuJoCo Sim2Sim

```bash
python deploy/deploy_mujoco/deploy_mujoco_pytorch.py x2.yaml
```

Test ONNX policy:

```bash
python deploy/deploy_mujoco/deploy_mujoco_onnx.py x2.yaml
```

Record video:

```bash
python deploy/deploy_mujoco/deploy_mujoco_pytorch.py x2.yaml --record
```

## 4.4 Monitoring (TensorBoard)

```bash
tensorboard --logdir ~/Projects/Legged_Gym/logs --bind_all
```

If TensorBoard dependencies are missing in your environment, install TensorFlow first.

## 4.5 Remote Port Forwarding Example

```bash
ssh -L 16006:amax:6006 2025-Fu.sc@125.217.226.174 -p 5983
ssh -L 16006:amax-Super-Server:6006 2025-Fu.sc@125.217.226.174 -p 5981
```

## 5. CLI Arguments Explained

The shared parser used by `train.py` and `play.py` includes these key arguments:

- `--task`
- `--resume`
- `--experiment_name`
- `--run_name`
- `--load_run`
- `--checkpoint`
- `--seed`
- `--num_envs`
- `--max_iterations`
- `--headless`
- `--sim_device`
- `--rl_device`

## 5.1 Core Selection Arguments

### `--task`

Selects robot task. For this branch, use:

```bash
--task x2
```

### `--experiment_name`

Top-level folder under `logs/`.

Example:

```bash
--experiment_name x2_reward_v2
```

### `--run_name`

Label for the new run in current launch.

Example:

```bash
--run_name seed1_debug
```

### `--load_run`

Selects existing run folder to load from (mainly used with `--resume` or `play.py`).

Example:

```bash
--load_run Mar11_14-22-10_seed1_debug
```

### `--checkpoint`

Selects checkpoint file (for example `model_3000.pt`).
If omitted or `-1`, latest checkpoint is used.

## 5.2 Training Control Arguments

### `--resume`

Resume from an existing run/checkpoint before continuing training.

### `--seed`

Controls random number generation (Python/NumPy/PyTorch/CUDA).

- fixed value (for example `1`) for reproducibility
- `-1` for random seed

### `--num_envs`

Parallel environments for training speed/throughput tradeoff.

### `--max_iterations`

Maximum PPO iterations for this launch.

### `--headless`

Disable rendering for higher training efficiency.

### `--sim_device` and `--rl_device`

Control simulation and RL compute devices.

## 6. Logs, Checkpoints, and Export Paths

Training output pattern:

```text
logs/<experiment_name>/<timestamp>_<run_name>/model_<iteration>.pt
```

Policy export path (during play):

```text
logs/<experiment_name>/exported/policies/
```

Naming recommendation:

```text
experiment_name = x2_reward_v2
run_name = seed1_debug
```
