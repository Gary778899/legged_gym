from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class X2FullbodyCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.67]  # x,y,z [m]
        default_joint_angles = {  # target angles [rad] when action = 0.0
            # lower body
            "left_hip_pitch_joint": -0.248,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.5303,
            "left_ankle_pitch_joint": -0.2823,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.248,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.5303,
            "right_ankle_pitch_joint": -0.2823,
            "right_ankle_roll_joint": 0.0,
            # upper body
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            # lower body
            "hip_yaw_joint": 120.0,
            "hip_roll_joint": 120.0,
            "hip_pitch_joint": 120.0,
            "knee_joint": 150.0,
            "ankle_pitch_joint": 40.0,
            "ankle_roll_joint": 30.0,
            # upper body
            "waist_yaw_joint": 160.0,
            "waist_pitch_joint": 80.0,
            "waist_roll_joint": 80.0,
            "shoulder_pitch_joint": 80.0,
            "shoulder_roll_joint": 40.0,
            "shoulder_yaw_joint": 40.0,
            "elbow_joint": 40.0,
            "wrist_yaw_joint": 40.0,
            "wrist_pitch_joint": 40.0,
            "wrist_roll_joint": 40.0,
            "head_yaw_joint": 20.0,
            "head_pitch_joint": 20.0,
        }
        damping = {
            # lower body
            "hip_yaw_joint": 5.0,
            "hip_roll_joint": 5.0,
            "hip_pitch_joint": 5.0,
            "knee_joint": 5.0,
            "ankle_pitch_joint": 3.0,
            "ankle_roll_joint": 2.0,
            # upper body
            "waist_yaw_joint": 5.0,
            "waist_pitch_joint": 5.0,
            "waist_roll_joint": 5.0,
            "shoulder_pitch_joint": 4.0,
            "shoulder_roll_joint": 1.0,
            "shoulder_yaw_joint": 1.0,
            "elbow_joint": 1.0,
            "wrist_yaw_joint": 1.0,
            "wrist_pitch_joint": 1.0,
            "wrist_roll_joint": 1.0,
            "head_yaw_joint": 2.0,
            "head_pitch_joint": 2.0,
        }
        # action scale: target angle = action_scale * action + default_angle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/urdf/x2_fullbody.urdf"
        name = "x2_fullbody"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.65
        command_threshold = 0.1
        yaw_command_threshold = 0.3

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.07
            orientation = -1.2
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -0.001
            feet_air_time = 0.2
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.2
            contact_no_vel = -0.25
            feet_swing_height = -15.0
            contact = 0.0
            symmetry_gait = 0.0
            stand_still = -0.0


class X2FullbodyCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 2000
        run_name = ""
        experiment_name = "x2_fullbody"
