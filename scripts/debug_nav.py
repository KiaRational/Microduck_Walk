import argparse, sys
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser); args_cli, h = parser.parse_known_args(); sys.argv = [sys.argv[0]] + h
import gymnasium as gym, torch
import microduck_lab.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config
env_cfg, _ = resolve_task_config("Microduck-Navigation", ""); env_cfg.scene.num_envs = 256
with launch_simulation(env_cfg, args_cli):
    env = gym.make("Microduck-Navigation", cfg=env_cfg); u = env.unwrapped; env.reset()
    r = u.scene["robot"]; t = u.scene.terrain
    print("[dbg] terrain origins z range", t.terrain_origins[..., 2].min().item(), t.terrain_origins[..., 2].max().item())
    print("[dbg] env origins", t.env_origins[:4].tolist())
    print("[dbg] root after reset", r.data.root_pos_w[:4].tolist(), "default root z", r.data.default_root_state[0, 2].item())
    for k in range(3):
        env.step(torch.zeros(u.num_envs, 3, device=u.device))
        print(f"[dbg] step {k} root z", [round(v, 3) for v in r.data.root_pos_w[:4, 2].tolist()], "term", u.termination_manager.dones[:4].tolist())
