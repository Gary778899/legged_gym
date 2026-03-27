import os
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# Monkeypatch external runner load to map CUDA storages to an available device.
try:
    import rsl_rl.runners.on_policy_runner as _opr
    _orig_load = getattr(_opr.OnPolicyRunner, 'load', None)
    def _patched_load(self, path, load_optimizer=True):
        try:
            target_device = torch.device(self.device)
        except Exception:
            target_device = torch.device('cpu')

        if target_device.type == 'cpu':
            map_location = 'cpu'
        else:
            def map_location(storage, loc):
                if isinstance(loc, str) and loc.startswith('cuda'):
                    return storage.cuda(target_device)
                return storage

        # Load with map_location to avoid errors when the saved checkpoint references
        # CUDA device indices that don't exist on this machine.
        try:
            loaded_dict = torch.load(path, map_location=map_location)
        except TypeError:
            # Older torch may not accept our callable; fall back to default load
            loaded_dict = torch.load(path)

        if 'model_state_dict' in loaded_dict:
            self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            except Exception:
                pass
        self.current_learning_iteration = loaded_dict.get('iter', getattr(self, 'current_learning_iteration', 0))
        return loaded_dict.get('infos', None)

    if _orig_load is not None:
        _opr.OnPolicyRunner.load = _patched_load
except Exception:
    pass


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', args.load_run + 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
