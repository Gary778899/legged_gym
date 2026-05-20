#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from aimdk_msgs.msg import JointStateArray
from core.constants import JOINT_ORDER
from core.observation_builder import ObservationBuilder
from core.onnx_policy_runner import OnnxPolicyRunner
from core.robot_state import RobotState, ScalingValues

class ControlMiddleware(Node):
    def __init__(self):
        super().__init__('control_middleware')
        self.state = RobotState()
        self.scales = ScalingValues()
        
        # Mapping and default pose array
        self.joint_name_to_index = {name: i for i, name in enumerate(JOINT_ORDER)}
        self.default_dof_pos = self.state.default_dof_pos()
        self.observation_builder = ObservationBuilder(
            default_dof_pos=self.default_dof_pos,
            scales=self.scales,
        )
        
        # Policy inputs
        self.commands = np.zeros(3) # x_vel, y_vel, yaw_vel
        self.phase = 0.0
        
        self.print_counter = 0 # Initialize the counter
        self.imu_ready = False
        self.joints_ready = False

        self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(JointStateArray, '/aima/hal/joint/leg/state', self.joint_state_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(0.02, self._control_loop)
        self.get_logger().info("Middleware started. Building observations...")

        # Load ONNX policy
        self.policy_runner = OnnxPolicyRunner(
            model_path='wbc_middleware/policy.onnx',
            hidden_dim=64,
            num_envs=1,
        )
        self.reset_lstm_memory()
        self.get_logger().info("ONNX policy loaded and LSTM states initialized.")

    def imu_callback(self, msg):
        self.state.quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.state.gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_ready = True

    def joint_state_callback(self, msg: JointStateArray):
        for joint_data in msg.joints:
            if joint_data.name in self.joint_name_to_index:
                idx = self.joint_name_to_index[joint_data.name]
                self.state.joint_pos[idx] = joint_data.position
                self.state.joint_vel[idx] = joint_data.velocity
        self.joints_ready = True

    def _control_loop(self):
        # if not (self.imu_ready and self.joints_ready):
        #     return
        
        # Update your phase here (example: 50Hz step)
        self.phase = (self.phase + 0.02) % (2 * np.pi)
        
        # Build the vector
        obs = self.build_observations()

        action = self.policy_runner.run(obs)
        self.state.last_action = action # Store last action for next obs

        # 5. Publish Command (Optional for now)
        # self.publish_action(action.flatten())

        # Handle the 1-second print (50Hz * 1s = 50 steps)
        self.print_counter += 1
        if self.print_counter >= 50:
            # Format the 47-dim array for readability
            obs_list = obs.flatten().tolist()
            formatted_obs = ", ".join([f"{x:.3f}" for x in obs_list])
            
            self.get_logger().info(f"\n--- FULL OBSERVATION (47 Dims) ---\n{formatted_obs}\n----------------------------------")
            
            self.print_counter = 0 # Reset the counter        

    def build_observations(self):
        return self.observation_builder.build(
            state=self.state,
            commands=self.commands,
            phase=self.phase,
        )
    
    def reset_lstm_memory(self):
        self.policy_runner.reset_memory()
        self.state.last_action.fill(0.0)
        self.get_logger().info("LSTM memory reset to zeros.")

def main():
    rclpy.init()
    node = ControlMiddleware()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
