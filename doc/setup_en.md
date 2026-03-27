# Installation Guide (X2 Extension)

## System Requirements

- Operating System: Ubuntu 18.04 or later
- GPU: NVIDIA GPU
- Driver Version: 525 or later recommended

## 1. Create a Virtual Environment

```bash
conda create -n x2-rl python=3.8
conda activate x2-rl
```

## 2. Install Dependencies

### 2.1 Install PyTorch

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# or
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2.2 Install Isaac Gym

1. Download Isaac Gym from NVIDIA.
2. Install:

```bash
cd isaacgym/python
pip install -e .
```

3. Verify:

```bash
cd examples
python 1080_balls_of_solitude.py
```

### 2.3 Install rsl_rl

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e .
```

### 2.4 Install This Repository

```bash
cd legged_gym
pip install -e .
```

### 2.5 Optional: Install a Robot SDK Python Package

If you deploy to physical hardware, install the required SDK package for your robot platform.

```bash
git clone <your_robot_sdk_repo>
cd <your_robot_sdk_dir>
pip install -e .
```

## Summary

After these steps, run X2 workflows from `doc/X2_GUIDE.md`.
