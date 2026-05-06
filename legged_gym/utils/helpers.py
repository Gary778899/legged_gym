import os
import copy
import re
import numpy as np
import random
import sys
import json
import socket
import platform
import getpass
from datetime import datetime
import subprocess
from typing import Optional, Any, Dict
import enum
from omegaconf import OmegaConf, DictConfig, ListConfig
from isaacgym import gymapi
from isaacgym import gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def update_class_from_omegaconf(cfg_class, dict_cfg: DictConfig):
    for key, value in dict_cfg.items():
        if hasattr(cfg_class, key):
            orig_attr = getattr(cfg_class, key)
            if isinstance(value, DictConfig):
                if isinstance(orig_attr, dict):
                    orig_attr.update(OmegaConf.to_container(value, resolve=True))
                elif isinstance(orig_attr, type) or hasattr(orig_attr, "__dict__"):
                    update_class_from_omegaconf(orig_attr, value)
                else:
                    setattr(cfg_class, key, OmegaConf.to_container(value, resolve=True))
            elif isinstance(value, ListConfig):
                setattr(cfg_class, key, OmegaConf.to_container(value, resolve=True))
            else:
                setattr(cfg_class, key, value)
        else:
            cfg_name = cfg_class.__name__ if hasattr(cfg_class, "__name__") else type(cfg_class).__name__
            print(f"[Warning] Key '{key}' not found in configuration class {cfg_name}. Ignoring.")

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)

    if load_run==-1:
        load_run_path = last_run
    else:
        # Support both run-name inputs (relative to root) and explicit checkpoint paths.
        load_run_path = load_run if os.path.isabs(load_run) else os.path.join(root, load_run)

    # If an explicit file is provided, use it directly.
    if os.path.isfile(load_run_path):
        return load_run_path

    if not os.path.isdir(load_run_path):
        raise FileNotFoundError(f"Run path does not exist or is not a directory: {load_run_path}")

    if checkpoint==-1:
        # Only consider torch checkpoints, not ONNX files or metadata.
        model_pattern = re.compile(r"^model_(\d+)\.pt$")
        models = [file for file in os.listdir(load_run_path) if model_pattern.match(file)]
        if not models:
            raise ValueError(f"No .pt checkpoints found in run directory: {load_run_path}")
        model = max(models, key=lambda m: int(model_pattern.match(m).group(1)))
    else:
        model = "model_{}.pt".format(checkpoint)
        model_path = os.path.join(load_run_path, model)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Requested checkpoint not found: {model_path}")

    load_path = os.path.join(load_run_path, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if hasattr(env_cfg, "domain_rand"):
            # CLI policy: domain randomization is disabled by default
            enable_domain_rand = bool(getattr(args, "domain_rand", False))
            if hasattr(env_cfg.domain_rand, "randomize_friction"):
                env_cfg.domain_rand.randomize_friction = enable_domain_rand
            if hasattr(env_cfg.domain_rand, "randomize_base_mass"):
                env_cfg.domain_rand.randomize_base_mass = enable_domain_rand
            if hasattr(env_cfg.domain_rand, "push_robots"):
                env_cfg.domain_rand.push_robots = enable_domain_rand
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    omega_args = getattr(args, "omegaconf_overrides", None)
    if omega_args is None:
        omega_args = [arg for arg in sys.argv[1:] if ('=' in arg and not arg.startswith('-'))]
    if omega_args:
        cli_conf = OmegaConf.from_cli(omega_args)
        if env_cfg is not None:
            env_overrides = {k: v for k, v in cli_conf.items() if hasattr(env_cfg, k)}
            if env_overrides:
                update_class_from_omegaconf(env_cfg, OmegaConf.create(env_overrides))
        if cfg_train is not None:
            train_overrides = {k: v for k, v in cli_conf.items() if hasattr(cfg_train, k)}
            if train_overrides:
                update_class_from_omegaconf(cfg_train, OmegaConf.create(train_overrides))

        unknown_keys = [
            k for k in cli_conf.keys()
            if (env_cfg is None or not hasattr(env_cfg, k)) and (cfg_train is None or not hasattr(cfg_train, k))
        ]
        for key in unknown_keys:
            print(f"[Warning] OmegaConf override root key '{key}' did not match env_cfg or cfg_train. Ignoring.")

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "x2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--domain_rand", "action": "store_true", "default": False, "help": "Enable domain randomization overrides."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    import argparse
    old_parse_args = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, args=None, namespace=None: self.parse_known_args(args, namespace)[0]
    try:
        args = gymutil.parse_arguments(
            description="RL Policy",
            custom_parameters=custom_parameters)
    finally:
        argparse.ArgumentParser.parse_args = old_parse_args

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"

    omega_args = [arg for arg in sys.argv[1:] if ('=' in arg and not arg.startswith('-'))]
    args.omegaconf_overrides = omega_args
    if omega_args:
        print(f"[OmegaConf] CLI overrides: {' '.join(omega_args)}")

    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class OnnxPolicyExporterLSTM(torch.nn.Module):
    """State-explicit LSTM wrapper for ONNX export.
    
    Exports recurrent policy with explicit hidden/cell state I/O for stateful inference.
    Note: ONNX LSTM export requires batch_size=1; if deploying with other batch sizes,
    handle input reshaping on the deployment side or export actor network only.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()

    def forward(self, obs, hidden_state, cell_state):
        """Forward pass with explicit state I/O for ONNX compatibility.
        
        Args:
            obs: (batch_size=1, num_obs) observation tensor
            hidden_state: (num_layers, batch_size=1, hidden_size) LSTM hidden state
            cell_state: (num_layers, batch_size=1, hidden_size) LSTM cell state
            
        Returns:
            action: (batch_size=1, num_actions) policy action
            next_hidden_state: updated hidden state
            next_cell_state: updated cell state
        """
        # LSTM expects input as (seq_len, batch, input_size).
        # Keep hidden/cell as 3D (num_layers, batch, hidden_size), so obs must be 3D too.
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0)
        elif obs.dim() != 3:
            raise ValueError(f"Expected obs to be 1D/2D/3D tensor, got shape {tuple(obs.shape)}")

        out, (next_hidden_state, next_cell_state) = self.memory(obs, (hidden_state, cell_state))
        action = self.actor(out.squeeze(0))
        return action, next_hidden_state, next_cell_state


def _policy_input_dim(actor_critic) -> int:
    actor = actor_critic.actor
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    raise ValueError("Could not infer actor input dimension for ONNX export")


def _to_metadata_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value)
    return json.dumps(_sanitize_for_serialization(value), sort_keys=True)


def _write_onnx_metadata(onnx_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import onnx
    except Exception as exc:
        return {
            "written": False,
            "reason": f"onnx package unavailable: {exc}",
        }

    model = onnx.load(onnx_path)
    merged = {prop.key: prop.value for prop in model.metadata_props}
    for key, value in metadata.items():
        merged[str(key)] = _to_metadata_value(value)

    del model.metadata_props[:]
    for key in sorted(merged.keys()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = merged[key]

    onnx.save(model, onnx_path)
    return {"written": True, "reason": ""}


def build_onnx_policy_metadata(env_cfg=None, train_cfg=None, args=None, checkpoint_file: Optional[str] = None):
    metadata = {
        "schema_version": "1.0.0",
        "validation_only": True,
        "runtime_source_of_truth": "deployment_yaml",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if args is not None:
        metadata["task"] = getattr(args, "task", None)
        metadata["rl_device"] = getattr(args, "rl_device", None)

    if checkpoint_file is not None:
        metadata["checkpoint_file"] = checkpoint_file

    if env_cfg is not None:
        env = getattr(env_cfg, "env", None)
        control = getattr(env_cfg, "control", None)
        normalization = getattr(env_cfg, "normalization", None)

        if env is not None:
            metadata["num_observations"] = getattr(env, "num_observations", None)
            metadata["num_actions"] = getattr(env, "num_actions", None)

        if control is not None:
            metadata["control"] = {
                "control_type": getattr(control, "control_type", None),
                "action_scale": getattr(control, "action_scale", None),
                "decimation": getattr(control, "decimation", None),
                # Keep kp/kd for validation against deploy YAML.
                "kp": getattr(control, "stiffness", None),
                "kd": getattr(control, "damping", None),
            }

        if normalization is not None:
            metadata["normalization"] = {
                "obs_scales": class_to_dict(getattr(normalization, "obs_scales", None)),
                "clip_observations": getattr(normalization, "clip_observations", None),
                "clip_actions": getattr(normalization, "clip_actions", None),
            }

    if train_cfg is not None:
        runner = getattr(train_cfg, "runner", None)
        policy = getattr(train_cfg, "policy", None)
        if runner is not None:
            metadata["runner"] = {
                "experiment_name": getattr(runner, "experiment_name", None),
                "run_name": getattr(runner, "run_name", None),
                "save_interval": getattr(runner, "save_interval", None),
                "max_iterations": getattr(runner, "max_iterations", None),
            }
        if policy is not None:
            metadata["policy"] = {
                "rnn_type": getattr(policy, "rnn_type", None),
                "rnn_hidden_size": getattr(policy, "rnn_hidden_size", None),
                "rnn_num_layers": getattr(policy, "rnn_num_layers", None),
            }

    return _sanitize_for_serialization(metadata)


def export_policy_as_onnx(
        actor_critic,
        onnx_path: str,
        num_obs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        write_sidecar: bool = True,
        opset_version: int = 17,
):
    """Export actor policy to ONNX and attach validation metadata.

    Metadata intentionally does not override runtime config; it is for consistency checks.
    """
    onnx_path = str(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if num_obs is None:
        num_obs = _policy_input_dim(actor_critic)
    num_obs = int(num_obs)

    is_recurrent = hasattr(actor_critic, "memory_a") and hasattr(actor_critic.memory_a, "rnn")
    if is_recurrent:
        exporter = OnnxPolicyExporterLSTM(actor_critic).to("cpu")
        exporter.eval()
        hidden_size = int(exporter.memory.hidden_size)
        num_layers = int(exporter.memory.num_layers)
        dummy_obs = torch.zeros(1, num_obs, dtype=torch.float32)
        dummy_hidden = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
        dummy_cell = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)

        import warnings
        # LSTM batch_size warning is safe: we export with batch=1 and explicit state I/O.
        # Suppress to reduce output noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*batch_size other than 1.*LSTM.*")
            torch.onnx.export(
                exporter,
                (dummy_obs, dummy_hidden, dummy_cell),
                onnx_path,
                export_params=True,
                do_constant_folding=True,
                opset_version=opset_version,
                input_names=["obs", "hidden_state", "cell_state"],
                output_names=["actions", "next_hidden_state", "next_cell_state"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "actions": {0: "batch"},
                    "hidden_state": {1: "batch"},
                    "cell_state": {1: "batch"},
                    "next_hidden_state": {1: "batch"},
                    "next_cell_state": {1: "batch"},
                },
            )
    else:
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        model.eval()
        dummy_obs = torch.zeros(1, num_obs, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_obs,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            opset_version=opset_version,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch"},
                "actions": {0: "batch"},
            },
        )
        print(f"[ONNX] Feedforward policy exported with batch dimension support.")

    metadata_payload = dict(metadata or {})
    metadata_payload["policy_is_recurrent"] = bool(is_recurrent)
    metadata_payload["policy_class"] = "ActorCriticRecurrent" if is_recurrent else "ActorCritic"
    metadata_payload["onnx_opset_version"] = int(opset_version)
    metadata_result = _write_onnx_metadata(onnx_path, metadata_payload)

    if write_sidecar:
        manifest = {
            "onnx_file": os.path.basename(onnx_path),
            "metadata_embedded": metadata_result["written"],
            "metadata_embed_reason": metadata_result["reason"],
            "metadata": metadata_payload,
        }
        with open(onnx_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_serialization(manifest), f, indent=2, ensure_ascii=False)

    if not metadata_result["written"]:
        print(f"[Warning] ONNX metadata was not embedded for {onnx_path}: {metadata_result['reason']}")
    
    if is_recurrent:
        print(f"[ONNX] Recurrent policy (LSTM) exported. Deploy with batch_size=1 or reshape on inference side.")


def write_run_onnx_metadata_once(log_dir: Optional[str], metadata: Dict[str, Any], opset_version: int) -> None:
    """Write one shared ONNX metadata sidecar per run directory."""
    if not log_dir:
        return

    # install_training_onnx_export_hook can run before save_training_config,
    # so ensure the run directory exists before writing metadata.
    os.makedirs(log_dir, exist_ok=True)

    shared_metadata_path = os.path.join(log_dir, "onnx_metadata.json")
    if os.path.exists(shared_metadata_path):
        return

    payload = dict(metadata or {})
    payload["onnx_opset_version"] = int(opset_version)
    payload["checkpoint_file"] = None

    manifest = {
        "metadata_embedded": False,
        "metadata_embed_reason": "run-level shared metadata file",
        "metadata": payload,
    }
    with open(shared_metadata_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_serialization(manifest), f, indent=2, ensure_ascii=False)


def install_training_onnx_export_hook(
        ppo_runner,
        env_cfg,
        train_cfg,
        args,
        opset_version: int = 17,
) -> bool:
    """Wrap runner checkpoint save so each model_N.pt emits a paired model_N.onnx."""
    if ppo_runner is None or not hasattr(ppo_runner, "save"):
        return False
    if getattr(ppo_runner, "_onnx_export_hook_installed", False):
        return True

    original_save = ppo_runner.save
    num_obs = int(getattr(env_cfg.env, "num_observations"))
    run_metadata = build_onnx_policy_metadata(
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        args=args,
        checkpoint_file=None,
    )
    write_run_onnx_metadata_once(getattr(ppo_runner, "log_dir", None), run_metadata, opset_version)

    def _save_with_onnx(path, *args_, **kwargs_):
        result = original_save(path, *args_, **kwargs_)

        try:
            checkpoint_path = str(path)
            checkpoint_file = os.path.basename(checkpoint_path)
            onnx_path = os.path.splitext(checkpoint_path)[0] + ".onnx"

            metadata = build_onnx_policy_metadata(
                env_cfg=env_cfg,
                train_cfg=train_cfg,
                args=args,
                checkpoint_file=checkpoint_file,
            )

            export_policy_as_onnx(
                actor_critic=ppo_runner.alg.actor_critic,
                onnx_path=onnx_path,
                num_obs=num_obs,
                metadata=metadata,
                write_sidecar=False,
                opset_version=opset_version,
            )
            print(f"[ONNX] Exported {onnx_path} alongside {checkpoint_file}")
        except Exception as exc:
            print(f"[Warning] Failed to export ONNX alongside checkpoint {path}: {exc}")

        return result

    ppo_runner.save = _save_with_onnx
    ppo_runner._onnx_export_hook_installed = True
    print("[ONNX] Training checkpoint export hook installed")
    return True


def _try_get_git_info(cwd: str):
    def _run(cmd):
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain"])
        dirty = len(status) > 0
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return None


def _sanitize_for_serialization(obj):
    """Convert objects to JSON/YAML friendly primitives.

    Isaac Gym args/configs can contain non-serializable objects (e.g. SimType enums).
    This makes dumps robust by converting to basic Python types.
    """
    if obj is None:
        return None

    # Simple primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Enum-like values (e.g. gymapi.SimType)
    if isinstance(obj, enum.Enum):
        return obj.name

    # Containers
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_serialization(v) for v in obj]

    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

    # torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)

    # argparse Namespace or similar
    if hasattr(obj, "__dict__"):
        try:
            return _sanitize_for_serialization(vars(obj))
        except Exception:
            pass

    # Fallback
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def save_training_config(
        log_dir: str,
        env_cfg=None,
        train_cfg=None,
        args=None,
        extra: Optional[dict] = None,
        write_yaml: bool = False,
) -> None:
    """Persist the resolved training configuration into the run log directory.

    Writes:
      - config.json: machine-friendly config dump
      - cmd.txt: the exact command used

        Notes:
            - JSON is the source of truth.
            - YAML is optional (opt-in via write_yaml=True).

    Safe to call even when log_dir is None.
    """
    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)
    cwd = os.getcwd()
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": cwd,
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "git": _try_get_git_info(cwd),
    }

    cfg_dump = {
        "meta": meta,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "env_cfg": class_to_dict(env_cfg) if env_cfg is not None else None,
        "train_cfg": class_to_dict(train_cfg) if train_cfg is not None else None,
    }
    if extra:
        cfg_dump["extra"] = extra

    cfg_dump = _sanitize_for_serialization(cfg_dump)

    # Write exact command line for quick copy/paste.
    try:
        with open(os.path.join(log_dir, "cmd.txt"), "w", encoding="utf-8") as f:
            f.write(meta["command"] + "\n")
    except Exception:
        pass

    # JSON (always available)
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    # YAML (optional)
    if write_yaml:
        try:
            import yaml

            yaml_text = yaml.safe_dump(cfg_dump, sort_keys=False, allow_unicode=True)
            with open(os.path.join(log_dir, "config.yaml"), "w", encoding="utf-8") as f:
                f.write(yaml_text)
        except Exception:
            # YAML is optional; JSON is the source of truth.
            pass


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
