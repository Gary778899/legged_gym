from setuptools import find_packages
from distutils.core import setup

setup(name='legged_gym_x2_ext',
      version='1.0.0',
      author='Sichao Fu',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='',
      description='X2-focused reinforcement learning environments and sim2sim tools',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.23.5', 'tensorboard', 'mujoco==3.2.3', 'pyyaml', 'protobuf<5.0.0', 'opencv-python', 'omegaconf', 'onnx>=1.14.0', 'onnxruntime>=1.14.0'])
