"""Open the obstacle navigation env in the GUI. Until a navigation policy is trained, each robot is driven
by a naive "turn toward the goal and walk" command (no obstacle avoidance) on top of the walking policy.

    ../../v6/run.sh scripts/view_nav.py --viz kit --seconds 60 [--video out.mp4]
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Microduck-Navigation-Play")
parser.add_argument("--seconds", type=float, default=60.0)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import microduck_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        print(f"[nav] {u.num_envs} robots, walking policy: {env_cfg.actions.walk.policy_path}", flush=True)
        reached = hits = 0
        for _ in range(int(args_cli.seconds / u.step_dt)):
            if not u.sim.is_running():
                break
            goal = u.command_manager.get_command("goal")[:, :2]  # goal in the base frame
            yaw_err = torch.atan2(goal[:, 1], goal[:, 0])
            act = torch.stack([(goal[:, 0] * 1.5).clamp(-1, 1) * (yaw_err.abs() < 0.6), torch.zeros_like(yaw_err),
                               (yaw_err * 1.5).clamp(-1, 1)], dim=1)
            _, _, term, trunc, _ = env.step(act)
            hits += int(u.termination_manager.get_term("hit_obstacle").sum().item())
            reached += int((goal.norm(dim=1) < 0.15).sum().item())
        depth = u.scene["depth_cam"].data.output["distance_to_image_plane"][0, ..., 0]
        print(f"[nav] obstacle hits: {hits}; depth image {tuple(depth.shape)}, range {depth.min().item():.2f}-{depth.max().item():.2f} m", flush=True)


if __name__ == "__main__":
    main()
