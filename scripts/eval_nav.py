"""Success rate of a navigation checkpoint: goal reached (within 0.15 m) before the episode ends.

    ../../v6/run.sh scripts/eval_nav.py --checkpoint logs/rsl_rl/microduck_nav/<run>/model_999.pt --num_envs 512
"""
import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Microduck-Navigation")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=1, help="episodes per robot")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    with launch_simulation(env_cfg, args_cli):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        n = u.num_envs
        done_eps = torch.zeros(n, dtype=torch.long, device=u.device)
        reached_now = torch.zeros(n, dtype=torch.bool, device=u.device)
        counts = {"reached": 0, "hit_obstacle": 0, "fell": 0, "timeout": 0}
        obs = env.get_observations()
        while (done_eps < args_cli.episodes).any():
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            dones = dones.bool()
            dist = u.command_manager.get_command("goal")[:, :2].norm(dim=1)
            reached_now |= dist < 0.15
            tm = u.termination_manager
            for i in torch.nonzero(dones & (done_eps < args_cli.episodes)).flatten().tolist():
                if reached_now[i]:
                    counts["reached"] += 1
                elif tm.get_term("hit_obstacle")[i]:
                    counts["hit_obstacle"] += 1
                elif tm.get_term("bad_orientation")[i] or tm.get_term("base_too_low")[i]:
                    counts["fell"] += 1
                else:
                    counts["timeout"] += 1
            done_eps += dones.long()
            reached_now &= ~dones
        total = sum(counts.values())
        print("[eval] " + ", ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in counts.items()) + f" of {total} episodes", flush=True)


if __name__ == "__main__":
    main()
