from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    'numpy==1.23.5',
    'mujoco==3.2.3',
    'pyyaml',
    'protobuf<5.0.0',
    'opencv-python',
    'omegaconf',
    'onnx>=1.14.0',
    'onnxruntime>=1.14.0',
]

EXTRAS_REQUIRE = {
    'train': [
        'isaacgym',
        'rsl-rl',
        'matplotlib',
        'tensorboard',
    ],
}


setup(
    name='legged_gym_x2_ext',
    version='1.0.0',
    author='Sichao Fu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='X2-focused reinforcement learning environments and sim2sim tools',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'middleware_ctl=wbc_middleware.ros2.middleware_ctl:main',
        ],
    },
)
