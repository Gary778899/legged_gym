import time
import glob
import os
from pathlib import Path

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

try:
    from deploy.deploy_mujoco.mujoco_logger import MujocoLogger
except ModuleNotFoundError:
    from mujoco_logger import MujocoLogger


CSV_ROOT_DIR = Path(LEGGED_GYM_ROOT_DIR) / "csv"


def get_gravity_orientation(quaternion):
    """Calculates the gravity vector in the robot's local frame from the base orientation quaternion."""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_policy_path(policy_path):
    if os.path.isfile(policy_path):
        return policy_path

    matching_paths = [path for path in glob.glob(policy_path) if os.path.isfile(path)]
    if matching_paths:
        return max(matching_paths, key=os.path.getmtime)

    raise ValueError(
        f"The provided filename {policy_path} does not exist, and no matching exported policy was found"
    )

def resolve_csv_output_path(filename: str) -> Path:
    output_name = Path(filename).name
    return CSV_ROOT_DIR / output_name

def as_float32_vector(values, expected_len):
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (expected_len,):
        raise ValueError(f"Expected a vector of length {expected_len}, got shape {vector.shape}")
    return vector


def get_startup_command_alpha(current_time_s, mode, hold_time_s, ramp_time_s):
    if mode == "zero_hold":
        return 0.0 if current_time_s < hold_time_s else 1.0
    if mode == "ramp_from_zero":
        if ramp_time_s <= 0.0:
            return 1.0
        return min(1.0, current_time_s / ramp_time_s)
    return 1.0

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--record", action="store_true", help="Record video from the track camera")
    parser.add_argument("--log_csv", action="store_true", help="Enable per-step MuJoCo logging to CSV")
    parser.add_argument("--log_output", type=str, default="mujoco_log.csv", help="Output CSV file name under csv/ (default: mujoco_log.csv)")
    parser.add_argument("--push-test", action="store_true", help="Enable the push test defined in the config")
    parser.add_argument("--push-time", type=float, default=None, help="Override push start time in seconds")
    parser.add_argument("--push-duration", type=float, default=None, help="Override push duration in seconds")
    parser.add_argument(
        "--push-force",
        type=float,
        nargs=3,
        metavar=("FX", "FY", "FZ"),
        default=None,
        help="Override the world-frame push force vector",
    )
    parser.add_argument("--camera", type=str, default="track", help="Camera name to use for recording")
    parser.add_argument("--output_file", type=str, default="recorded_video.mp4", help="Output video file (default: recorded_video.mp4)")
    parser.add_argument("--video_width", type=int, default=1920, help="Video width for recording (default: 1920)")
    parser.add_argument("--video_height", type=int, default=1080, help="Video height for recording (default: 1080)")
    parser.add_argument("--record_fps", type=int, default=30, help="Video FPS - records every Nth frame to match this (default: 30)")
    parser.add_argument(
        "--record_time_base",
        type=str,
        choices=["wall", "simulation"],
        default="wall",
        help="Use wall time to match the viewer playback, or simulation time to normalize playback speed (default: wall)",
    )
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = resolve_policy_path(
            config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        )
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        startup_command_config = config.get("startup_command", {})
        startup_command_mode = str(startup_command_config.get("mode", "disabled")).lower()
        startup_command_hold_time_s = float(startup_command_config.get("hold_time_s", 0.0))
        startup_command_ramp_time_s = float(startup_command_config.get("ramp_time_s", 0.0))

        valid_startup_modes = {"disabled", "zero_hold", "ramp_from_zero"}
        if startup_command_mode not in valid_startup_modes:
            raise ValueError(
                f"startup_command.mode must be one of {sorted(valid_startup_modes)}, got '{startup_command_mode}'"
            )
        if startup_command_hold_time_s < 0.0:
            raise ValueError("startup_command.hold_time_s must be nonnegative")
        if startup_command_ramp_time_s < 0.0:
            raise ValueError("startup_command.ramp_time_s must be nonnegative")

        if startup_command_mode == "zero_hold" and startup_command_ramp_time_s > 0.0:
            raise ValueError("startup_command.ramp_time_s must be 0 when mode is 'zero_hold'")
        if startup_command_mode == "ramp_from_zero" and startup_command_hold_time_s > 0.0:
            raise ValueError("startup_command.hold_time_s must be 0 when mode is 'ramp_from_zero'")

        # Push test parameters
        push_config = config.get("push_test", {})
        push_enabled = bool(push_config.get("enabled", False) or args.push_test)
        push_start_time_s = float(push_config.get("start_time_s", 10.0))
        push_duration_s = float(push_config.get("duration_s", 0.10))
        push_force_world = as_float32_vector(push_config.get("force_world", [0.0, 0.0, 0.0]), 3)
        push_body_name = str(push_config.get("body_name", "pelvis")) # Default to "pelvis" if not specified

        if args.push_time is not None:
            push_start_time_s = float(args.push_time)
        if args.push_duration is not None:
            push_duration_s = float(args.push_duration)
        if args.push_force is not None:
            push_force_world = as_float32_vector(args.push_force, 3)

        if push_enabled and push_duration_s <= 0.0:
            raise ValueError("push_test.duration_s must be greater than zero when push_test is enabled")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    push_body_id = None
    if push_enabled:
        push_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, push_body_name)
        if push_body_id < 0:
            raise ValueError(f"Push body '{push_body_name}' was not found in the MuJoCo model")

    # Set initial joint positions before starting the viewer
    d.qpos[7:7+num_actions] = default_angles
    mujoco.mj_forward(m, d) # Update kinematics for the visualizer

    # load policy
    policy = torch.jit.load(policy_path)

    # Setup frame recording if enabled
    if args.record:
        # Import cv2 only when recording is needed
        try:
            import cv2
        except ImportError:
            print("Error: OpenCV (cv2) is required for video recording. Install it with: pip install opencv-python")
            exit(1)

        record_interval = 1.0 / args.record_fps

        # Encode at the requested FPS and sample frames according to simulation time.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_file, fourcc, args.record_fps, (args.video_width, args.video_height))
        
        print(f"Recording video to: {args.output_file}")
        print(f"  Resolution: {args.video_width}x{args.video_height}")
        print(f"  Video FPS: {args.record_fps:.1f}")
        if args.record_time_base == "wall":
            print("  Time base: wall clock (video matches what is shown in the viewer)")
        else:
            print("  Time base: simulation (video normalizes out rendering slowdowns)")
        
        # Get camera ID
        try:
            camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        except:
            print(f"Warning: Camera '{args.camera}' not found, using default camera")
            camera_id = -1
        
        # Create offscreen renderer with specified resolution
        try:
            renderer = mujoco.Renderer(m, height=args.video_height, width=args.video_width)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"\nTip: If framebuffer is still too small, you can specify lower resolution with:")
            print(f"  python deploy/deploy_mujoco/deploy_mujoco.py x2.yaml --record --video_width 640 --video_height 480")
            exit(1)
        
        frame_count = 0
    else:
        renderer = None
        camera_id = None
        frame_count = 0
        out = None
        record_interval = 0.0

    # Setup logger if enabled
    logger = None
    if args.log_csv:
        logger = MujocoLogger(
            joint_names=[f"joint_{i}" for i in range(num_actions)],
            qpos_slice=slice(7, 7 + num_actions),
            qvel_slice=slice(6, 6 + num_actions),
            torque_slice=slice(0, num_actions),
        )


    with mujoco.viewer.launch_passive(m, d) as viewer:
        wall_start = time.time()
        sim_start = d.time
        if args.record_time_base == "wall":
            next_frame_time = wall_start
        else:
            next_frame_time = sim_start
        if args.record:
            print(
                f"Recording started. Max simulated duration: {simulation_duration}s. Close viewer to stop early."
            )
        if push_enabled:
            print(
                "Push test enabled: "
                f"body={push_body_name}, start_time={push_start_time_s:.3f}s, "
                f"duration={push_duration_s:.3f}s, force_world={push_force_world.tolist()}"
            )
        if startup_command_mode == "zero_hold":
            print(f"Startup command mode: zero_hold, hold_time={startup_command_hold_time_s:.3f}s")
        elif startup_command_mode == "ramp_from_zero":
            print(f"Startup command mode: ramp_from_zero, ramp_time={startup_command_ramp_time_s:.3f}s")
        else:
            print("Startup command mode: disabled")
        while viewer.is_running() and d.time - sim_start < simulation_duration:
            step_start = time.time()
            sim_time = d.time - sim_start
            push_active = push_enabled and (sim_time >= push_start_time_s) and (sim_time < push_start_time_s + push_duration_s)

            d.xfrc_applied[:] = 0.0
            if push_active and push_body_id is not None:
                d.xfrc_applied[push_body_id, :3] = push_force_world

            # Control all joints using PD controller
            tau = pd_control(target_dof_pos, d.qpos[7:7+num_actions], kps, np.zeros_like(kds), d.qvel[6:6+num_actions], kds)
            d.ctrl[:num_actions] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            if logger is not None:
                logger.log_step(d, push_active=push_active, push_force_world=push_force_world if push_active else None)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # Create observation - use all joints
                qj = d.qpos[7:7+num_actions]
                dqj = d.qvel[6:6+num_actions]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 1.0
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                current_time = counter * simulation_dt
                cmd_alpha = get_startup_command_alpha(
                    current_time,
                    startup_command_mode,
                    startup_command_hold_time_s,
                    startup_command_ramp_time_s,
                )

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = (cmd * cmd_alpha) * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # Warmup: Smoothly blend the action scale from 0 to 1 over the first 1 second
                warmup_time = 0. # seconds
                alpha = min(1.0, current_time / warmup_time) if warmup_time > 0 else 1.0
                # transform action to target_dof_pos with alpha
                target_dof_pos = (action * alpha) * action_scale + default_angles

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

            # Record frame if enabled
            if args.record and renderer is not None:
                current_record_time = time.time() if args.record_time_base == "wall" else d.time
                frames_to_write = 0
                while current_record_time + 1e-9 >= next_frame_time:
                    frames_to_write += 1
                    next_frame_time += record_interval

                if frames_to_write > 0:
                    renderer.update_scene(d, camera=camera_id)
                    pixels = renderer.render()
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    for _ in range(frames_to_write):
                        out.write(frame_bgr)
                    frame_count += frames_to_write

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # Calculate elapsed time in both wall clock and simulation time.
        elapsed_wall_time = time.time() - wall_start
        elapsed_sim_time = d.time - sim_start
        
        # Print recording summary and cleanup
        if args.record:
            # Release video writer
            out.release()
            
            print(f"\n{'='*60}")
            print(f"Video recording complete!")
            print(f"  Frames recorded: {frame_count}")
            print(f"  Output file: {args.output_file}")
            print(f"  Simulated time: {elapsed_sim_time:.2f}s / {simulation_duration}s")
            print(f"  Wall time: {elapsed_wall_time:.2f}s")
            print(f"  Video duration: {frame_count / args.record_fps:.2f}s at {args.record_fps:.1f} FPS")
            print(f"{'='*60}")
            print(f"\nThe video playback speed matches the simulation speed.")

        if logger is not None:
            CSV_ROOT_DIR.mkdir(parents=True, exist_ok=True)
            log_output_path = resolve_csv_output_path(args.log_output)
            logger.save_to_csv(str(log_output_path))
            print(f"MuJoCo log saved to: {log_output_path}")