"""Record many Microducks running a checkpoint side by side (headless), e.g. to compare training stages.

    ../../v6/run.sh scripts/record_policy.py --checkpoint logs/rsl_rl/microduck_flat/<run>/model_200.pt
    ../../v6/run.sh scripts/record_policy.py --checkpoint <ckpt> --num_envs 100 --seconds 12 --export

Writes <checkpoint dir>/videos/<checkpoint name>.mp4 from an overview camera above the grid of robots
(velocity commands are resampled every few seconds so the robots walk in different directions).
--export also writes exported/policy.pt (TorchScript, used as the low-level policy for navigation).
"""

import argparse
import math
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Microduck-Velocity-Flat-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--export", action="store_true")
parser.add_argument("--label", default="", help="text burned into the corner of the video")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import microduck_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def look_at_quat(eye, target):
    """xyzw quaternion for a camera at ``eye`` looking at ``target`` (world convention: +x forward, +z up)."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return tuple(Rotation.from_matrix(np.stack([f, left, up], axis=1)).as_quat().tolist())


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity.resampling_time_range = (3.0, 3.0)
    env_cfg.commands.base_velocity.debug_vis = False
    side = math.ceil(math.sqrt(args_cli.num_envs)) * env_cfg.scene.env_spacing
    eye, target = (side * 0.55, -side * 0.55, side * 0.42), (0.0, 0.0, 0.0)
    env_cfg.scene.overview = TiledCameraCfg(
        prim_path="/World/overview_cam", height=720, width=1280, data_types=["rgb"], update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, target), convention="world"),
    )
    with launch_simulation(env_cfg, args_cli):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        ckpt = os.path.abspath(args_cli.checkpoint)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        run_dir = os.path.dirname(ckpt)
        if args_cli.export:
            runner.export_policy_to_jit(path=os.path.join(run_dir, "exported"), filename="policy.pt")
            print(f"[record] exported {os.path.join(run_dir, 'exported', 'policy.pt')}")
        cam = env.unwrapped.scene["overview"]
        out_dir = os.path.join(run_dir, "videos")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(ckpt))[0] + ".mp4")
        fps = round(1.0 / env.unwrapped.step_dt)
        writer = iio.get_writer(out, fps=fps)
        obs = env.get_observations()
        falls = 0
        for _ in range(int(args_cli.seconds / env.unwrapped.step_dt)):
            with torch.inference_mode():
                obs, _, dones, extras = env.step(policy(obs))
            falls += int(dones.sum().item())
            frame = cam.data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
            if args_cli.label:
                import cv2

                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, args_cli.label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
            writer.append_data(frame)
        writer.close()
        print(f"[record] {args_cli.num_envs} robots, {args_cli.seconds:.0f} s, episode ends (falls/timeouts): {falls}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
