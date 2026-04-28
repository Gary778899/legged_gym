import glob
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

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
    """Calculates torques from position commands."""
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


def get_policy_path_for_backend(config, backend):
    if backend == "pytorch":
        raw_policy_path = config.get("policy_path_pt") or config.get("policy_path")
        if not raw_policy_path:
            raise ValueError("Missing policy path for PyTorch backend. Set policy_path_pt or policy_path in YAML.")
    elif backend == "onnx":
        raw_policy_path = config.get("policy_path_onnx")
        if not raw_policy_path:
            # Fallback to legacy key if user points directly to .onnx there.
            raw_policy_path = config.get("policy_path")
        if not raw_policy_path:
            raise ValueError("Missing policy path for ONNX backend. Set policy_path_onnx in YAML.")
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'pytorch' or 'onnx'.")

    policy_path = resolve_policy_path(
        raw_policy_path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    )
    return policy_path


def extract_metadata_payload(metadata):
    return metadata.get("metadata", {}) if isinstance(metadata, dict) else {}


def load_onnx_metadata(policy_path):
    import json

    result = {"warnings": [], "metadata": None}

    if str(policy_path).endswith(".onnx"):
        onnx_path = str(policy_path)
    else:
        onnx_path = str(policy_path).replace(".pt", ".onnx")

    shared_metadata_path = os.path.join(os.path.dirname(onnx_path), "onnx_metadata.json")
    metadata_sidecar = onnx_path + ".meta.json"
    metadata = None

    if os.path.exists(shared_metadata_path):
        try:
            with open(shared_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            result["warnings"].append(f"Failed to load shared ONNX metadata: {exc}")

    if metadata is None and os.path.exists(metadata_sidecar):
        try:
            with open(metadata_sidecar, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            result["warnings"].append(f"Failed to load ONNX sidecar metadata: {exc}")

    if metadata is None:
        result["warnings"].append(
            f"No ONNX metadata found at {shared_metadata_path} or {metadata_sidecar}. "
            "Skipping metadata validation. Ensure deployment config matches training config."
        )

    result["metadata"] = metadata
    return result


def validate_onnx_metadata(policy_path, yaml_config):
    """Validate ONNX model metadata against deployment config.

    Returns dict with:
        - validated: bool (all checks passed or no metadata found)
        - warnings: list of warning strings
        - errors: list of error strings (non-fatal)
    """
    result = {"validated": True, "warnings": [], "errors": []}

    load_result = load_onnx_metadata(policy_path)
    result["warnings"].extend(load_result["warnings"])
    metadata = load_result["metadata"]

    if metadata is None:
        return result

    metadata_payload = extract_metadata_payload(metadata)

    meta_num_obs = metadata_payload.get("num_observations")
    meta_num_actions = metadata_payload.get("num_actions")
    if meta_num_obs is not None and meta_num_obs != yaml_config.get("num_obs"):
        result["errors"].append(
            f"Observation dimension mismatch: metadata={meta_num_obs}, YAML={yaml_config.get('num_obs')}"
        )
    if meta_num_actions is not None and meta_num_actions != yaml_config.get("num_actions"):
        result["errors"].append(
            f"Action dimension mismatch: metadata={meta_num_actions}, YAML={yaml_config.get('num_actions')}"
        )

    meta_control = metadata_payload.get("control", {})
    meta_kp = meta_control.get("kp")
    meta_kd = meta_control.get("kd")

    def _check_numeric_list_compat(meta_values, yaml_values, label):
        if len(meta_values) == 0 or len(yaml_values) == 0:
            return

        # Common case: metadata stores grouped joint parameters (e.g., 6),
        # while deployment YAML stores per-joint values (e.g., 12).
        if len(meta_values) != len(yaml_values):
            if len(yaml_values) % len(meta_values) == 0:
                repeat_factor = len(yaml_values) // len(meta_values)
                expanded = np.repeat(np.asarray(meta_values, dtype=np.float32), repeat_factor)
                yaml_arr = np.asarray(yaml_values, dtype=np.float32)
                if not np.allclose(expanded, yaml_arr, rtol=1e-5):
                    # Deployment YAML often stores per-joint values in actuator order,
                    # while training metadata can store grouped values in joint-type order.
                    # Accept as compatible if value multisets match after expansion.
                    expanded_sorted = np.sort(expanded)
                    yaml_sorted = np.sort(yaml_arr)
                    if np.allclose(expanded_sorted, yaml_sorted, rtol=1e-5):
                        result["warnings"].append(
                            f"{label} comparison passed after order-insensitive grouped expansion "
                            f"({len(meta_values)} -> {len(yaml_values)})."
                        )
                    else:
                        result["errors"].append(
                            f"{label} mismatch after expanding grouped metadata values "
                            f"({len(meta_values)} -> {len(yaml_values)})."
                        )
            else:
                result["warnings"].append(
                    f"Skipping strict {label} comparison due to incompatible lengths: "
                    f"metadata={len(meta_values)}, YAML={len(yaml_values)}."
                )
            return

        meta_arr = np.asarray(meta_values, dtype=np.float32)
        yaml_arr = np.asarray(yaml_values, dtype=np.float32)
        if not np.allclose(meta_arr, yaml_arr, rtol=1e-5):
            result["errors"].append(
                f"{label} mismatch between metadata and YAML config."
            )

    if meta_kp is not None:
        yaml_kps = yaml_config.get("kps", [])
        if isinstance(meta_kp, dict):
            meta_kp_list = list(meta_kp.values())
        else:
            meta_kp_list = meta_kp if isinstance(meta_kp, list) else [meta_kp]

        _check_numeric_list_compat(
            meta_kp_list,
            yaml_kps,
            "Joint stiffness (kp)",
        )

    if meta_kd is not None:
        yaml_kds = yaml_config.get("kds", [])
        if isinstance(meta_kd, dict):
            meta_kd_list = list(meta_kd.values())
        else:
            meta_kd_list = meta_kd if isinstance(meta_kd, list) else [meta_kd]

        _check_numeric_list_compat(
            meta_kd_list,
            yaml_kds,
            "Joint damping (kd)",
        )

    meta_norm = metadata_payload.get("normalization", {})
    meta_obs_scales = meta_norm.get("obs_scales", {})

    yaml_ang_vel_scale = yaml_config.get("ang_vel_scale")
    meta_ang_vel_scale = meta_obs_scales.get("ang_vel")
    if meta_ang_vel_scale is not None and yaml_ang_vel_scale is not None:
        if not np.isclose(meta_ang_vel_scale, yaml_ang_vel_scale, rtol=1e-5):
            result["errors"].append(
                f"Angular velocity scale mismatch: metadata={meta_ang_vel_scale}, YAML={yaml_ang_vel_scale}"
            )

    yaml_dof_pos_scale = yaml_config.get("dof_pos_scale")
    meta_dof_pos_scale = meta_obs_scales.get("dof_pos")
    if meta_dof_pos_scale is not None and yaml_dof_pos_scale is not None:
        if not np.isclose(meta_dof_pos_scale, yaml_dof_pos_scale, rtol=1e-5):
            result["errors"].append(
                f"DOF position scale mismatch: metadata={meta_dof_pos_scale}, YAML={yaml_dof_pos_scale}"
            )

    yaml_dof_vel_scale = yaml_config.get("dof_vel_scale")
    meta_dof_vel_scale = meta_obs_scales.get("dof_vel")
    if meta_dof_vel_scale is not None and yaml_dof_vel_scale is not None:
        if not np.isclose(meta_dof_vel_scale, yaml_dof_vel_scale, rtol=1e-5):
            result["errors"].append(
                f"DOF velocity scale mismatch: metadata={meta_dof_vel_scale}, YAML={yaml_dof_vel_scale}"
            )

    yaml_action_scale = yaml_config.get("action_scale")
    meta_action_scale = meta_control.get("action_scale")
    if meta_action_scale is not None and yaml_action_scale is not None:
        if not np.isclose(meta_action_scale, yaml_action_scale, rtol=1e-5):
            result["errors"].append(
                f"Action scale mismatch: metadata={meta_action_scale}, YAML={yaml_action_scale}"
            )

    if result["errors"]:
        result["validated"] = False

    return result


class TorchPolicyRunner:
    def __init__(self, policy_path):
        self.policy = torch.jit.load(policy_path)

    def infer(self, obs):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        return self.policy(obs_tensor).detach().cpu().numpy().squeeze()


class OnnxPolicyRunner:
    def __init__(self, policy_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX deployment. Install it with 'pip install onnxruntime'."
            ) from exc

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

        if not self.input_names:
            raise ValueError("ONNX model has no inputs.")

        self.obs_input_name = "obs" if "obs" in self.input_names else self.input_names[0]
        self.recurrent_state = {}

        for inp in self.session.get_inputs():
            if inp.name == self.obs_input_name:
                continue
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    shape.append(1)
            self.recurrent_state[inp.name] = np.zeros(shape, dtype=np.float32)

        self.action_output_name = "actions" if "actions" in self.output_names else self.output_names[0]

    def infer(self, obs):
        feed_dict = {self.obs_input_name: obs[np.newaxis, :].astype(np.float32)}
        for state_name, state_value in self.recurrent_state.items():
            feed_dict[state_name] = state_value

        output_values = self.session.run(None, feed_dict)
        output_map = {name: value for name, value in zip(self.output_names, output_values)}

        for name, value in output_map.items():
            if name.startswith("next_"):
                candidate = name[len("next_") :]
                if candidate in self.recurrent_state:
                    self.recurrent_state[candidate] = value

        action = output_map[self.action_output_name]
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(f"Unexpected action output shape from ONNX policy: {action.shape}")
        return action.squeeze(0)


def load_policy_runner(backend, policy_path):
    if backend == "pytorch":
        if not str(policy_path).endswith(".pt"):
            print(f"[Warning] PyTorch backend expected a .pt file, got: {policy_path}")
        return TorchPolicyRunner(policy_path)

    if backend == "onnx":
        if not str(policy_path).endswith(".onnx"):
            raise ValueError(
                f"ONNX backend requires a .onnx file. Received: {policy_path}. "
                "Set policy_path_onnx in YAML to an ONNX model path."
            )
        return OnnxPolicyRunner(policy_path)

    raise ValueError(f"Unsupported backend '{backend}'.")


def parse_args():
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
    return parser.parse_args()


def run(backend):
    args = parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = get_policy_path_for_backend(config, backend)
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

        push_config = config.get("push_test", {})
        push_enabled = bool(push_config.get("enabled", False) or args.push_test)
        push_start_time_s = float(push_config.get("start_time_s", 10.0))
        push_duration_s = float(push_config.get("duration_s", 0.10))
        push_force_world = as_float32_vector(push_config.get("force_world", [0.0, 0.0, 0.0]), 3)
        push_body_name = str(push_config.get("body_name", "pelvis"))

        if args.push_time is not None:
            push_start_time_s = float(args.push_time)
        if args.push_duration is not None:
            push_duration_s = float(args.push_duration)
        if args.push_force is not None:
            push_force_world = as_float32_vector(args.push_force, 3)

        if push_enabled and push_duration_s <= 0.0:
            raise ValueError("push_test.duration_s must be greater than zero when push_test is enabled")

        print("\n[Deployment] Backend:", backend)
        print("[Deployment] Policy:", policy_path)

        if backend == "onnx":
            print("[Deployment] Validating ONNX policy metadata...")
            metadata_validation = validate_onnx_metadata(policy_path, config)
            if metadata_validation["warnings"]:
                for warning in metadata_validation["warnings"]:
                    print(f"[Warning] {warning}")
            if metadata_validation["errors"]:
                print("[Error] Model metadata validation failed with the following mismatches:")
                for error in metadata_validation["errors"]:
                    print(f"  - {error}")
                print("\nTo proceed despite mismatches, ensure YAML config matches training config.")
                print("Deployment will still proceed; however, policy behavior may differ from training.")
            else:
                print("[OK] ONNX metadata matches deployment configuration.")

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    push_body_id = None
    if push_enabled:
        push_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, push_body_name)
        if push_body_id < 0:
            raise ValueError(f"Push body '{push_body_name}' was not found in the MuJoCo model")

    d.qpos[7 : 7 + num_actions] = default_angles
    mujoco.mj_forward(m, d)

    policy_runner = load_policy_runner(backend, policy_path)

    if args.record:
        try:
            import cv2
        except ImportError:
            print("Error: OpenCV (cv2) is required for video recording. Install it with: pip install opencv-python")
            raise

        record_interval = 1.0 / args.record_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_file, fourcc, args.record_fps, (args.video_width, args.video_height))

        print(f"Recording video to: {args.output_file}")
        print(f"  Resolution: {args.video_width}x{args.video_height}")
        print(f"  Video FPS: {args.record_fps:.1f}")
        if args.record_time_base == "wall":
            print("  Time base: wall clock (video matches what is shown in the viewer)")
        else:
            print("  Time base: simulation (video normalizes out rendering slowdowns)")

        try:
            camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        except Exception:
            print(f"Warning: Camera '{args.camera}' not found, using default camera")
            camera_id = -1

        try:
            renderer = mujoco.Renderer(m, height=args.video_height, width=args.video_width)
        except ValueError as e:
            print(f"Error: {e}")
            print("\nTip: If framebuffer is still too small, you can specify lower resolution with:")
            print(
                "  python deploy/deploy_mujoco/deploy_mujoco_pytorch.py x2.yaml "
                "--record --video_width 640 --video_height 480"
            )
            raise

        frame_count = 0
    else:
        renderer = None
        camera_id = None
        frame_count = 0
        out = None
        record_interval = 0.0

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
            push_active = push_enabled and (sim_time >= push_start_time_s) and (
                sim_time < push_start_time_s + push_duration_s
            )

            d.xfrc_applied[:] = 0.0
            if push_active and push_body_id is not None:
                d.xfrc_applied[push_body_id, :3] = push_force_world

            tau = pd_control(
                target_dof_pos,
                d.qpos[7 : 7 + num_actions],
                kps,
                np.zeros_like(kds),
                d.qvel[6 : 6 + num_actions],
                kds,
            )
            d.ctrl[:num_actions] = tau
            mujoco.mj_step(m, d)
            if logger is not None:
                logger.log_step(d, push_active=push_active, push_force_world=push_force_world if push_active else None)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7 : 7 + num_actions]
                dqj = d.qvel[6 : 6 + num_actions]
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

                action = policy_runner.infer(obs).astype(np.float32)
                warmup_time = 0.0
                alpha = min(1.0, current_time / warmup_time) if warmup_time > 0 else 1.0
                target_dof_pos = (action * alpha) * action_scale + default_angles

                viewer.sync()

            if args.record and renderer is not None:
                current_record_time = time.time() if args.record_time_base == "wall" else d.time
                frames_to_write = 0
                while current_record_time + 1e-9 >= next_frame_time:
                    frames_to_write += 1
                    next_frame_time += record_interval

                if frames_to_write > 0:
                    renderer.update_scene(d, camera=camera_id)
                    pixels = renderer.render()

                    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    for _ in range(frames_to_write):
                        out.write(frame_bgr)
                    frame_count += frames_to_write

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        elapsed_wall_time = time.time() - wall_start
        elapsed_sim_time = d.time - sim_start

        if args.record:
            out.release()

            print(f"\n{'=' * 60}")
            print("Video recording complete!")
            print(f"  Frames recorded: {frame_count}")
            print(f"  Output file: {args.output_file}")
            print(f"  Simulated time: {elapsed_sim_time:.2f}s / {simulation_duration}s")
            print(f"  Wall time: {elapsed_wall_time:.2f}s")
            print(f"  Video duration: {frame_count / args.record_fps:.2f}s at {args.record_fps:.1f} FPS")
            print(f"{'=' * 60}")
            print("\nThe video playback speed matches the simulation speed.")

        if logger is not None:
            CSV_ROOT_DIR.mkdir(parents=True, exist_ok=True)
            log_output_path = resolve_csv_output_path(args.log_output)
            logger.save_to_csv(str(log_output_path))
            print(f"MuJoCo log saved to: {log_output_path}")
