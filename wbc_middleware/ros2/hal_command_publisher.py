from __future__ import annotations

from typing import Sequence

from aimdk_msgs.msg import JointCommand, JointCommandArray

from wbc_middleware.core.command_mapper import JointCommandTarget


class HalCommandPublisher:
    def __init__(self, node, topic_name: str):
        self._node = node
        self._publisher = node.create_publisher(JointCommandArray, topic_name, 10)
        self._sequence = 0

    def publish(self, targets: Sequence[JointCommandTarget]) -> None:
        message = JointCommandArray()
        now_msg = self._node.get_clock().now().to_msg()
        message.header.stamp = now_msg
        message.header.meas_stamp = now_msg
        message.header.sequence = self._sequence
        message.header.frame_id = "base"
        self._sequence += 1

        for target in targets:
            joint_command = JointCommand()
            joint_command.name = target.name
            joint_command.position = float(target.position)
            joint_command.velocity = float(target.velocity)
            joint_command.effort = float(target.effort)
            joint_command.stiffness = float(target.stiffness)
            joint_command.damping = float(target.damping)
            message.joints.append(joint_command)

        self._publisher.publish(message)
