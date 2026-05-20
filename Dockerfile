FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PROJECT_ROOT=/workspace/legged_gym
ENV ROS_WS=/workspace/.ros2_ws
ENV PYTHONPATH=/workspace/legged_gym

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    bash-completion \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxinerama1 \
    libxcursor1 \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --system-site-packages "${VIRTUAL_ENV}" \
    && uv pip install \
    numpy==1.23.5 \
    matplotlib \
    tensorboard \
    mujoco==3.2.3 \
    pyyaml \
    "protobuf<5.0.0" \
    opencv-python \
    omegaconf \
    "onnx>=1.14.0" \
    "onnxruntime>=1.14.0"

RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
    && echo 'if [ -f /workspace/.ros2_ws/install/setup.bash ]; then source /workspace/.ros2_ws/install/setup.bash; fi' >> /etc/bash.bashrc \
    && echo 'export PROJECT_ROOT=/workspace/legged_gym' >> /etc/bash.bashrc \
    && echo 'export ROS_WS=/workspace/.ros2_ws' >> /etc/bash.bashrc \
    && echo 'export PYTHONPATH=/workspace/legged_gym:${PYTHONPATH}' >> /etc/bash.bashrc

WORKDIR /workspace/legged_gym
CMD ["sleep", "infinity"]
