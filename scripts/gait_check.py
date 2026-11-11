"""Gait diagnostics for a walking checkpoint: fixed forward command, measure speed, foot lift and air time.

    ../../v6/run.sh scripts/gait_check.py --checkpoint <ckpt> --vx 0.3
"""
import argparse, os, sys
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--task", default="Microduck-Velocity-Flat-Play")
p.add_argument("--checkpoint", required=True)
p.add_argument("--vx", type=float, default=0.3)
p.add_argument("--num_envs", type=int, default=16)
add_launcher_args(p)
a, h = p.parse_known_args(); sys.argv = [sys.argv[0]] + h
import gymnasium as gym, torch  # noqa
import microduck_lab.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config  # noqa


def main():
    env_cfg, agent_cfg = resolve_task_config(a.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = a.num_envs
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (a.vx, a.vx); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
    c.rel_standing_envs = 0.0; c.resampling_time_range = (100.0, 100.0)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    with launch_simulation(env_cfg, a):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        r = create_rsl_rl_runner(env, agent_cfg); r.load(os.path.abspath(a.checkpoint))
        pol = r.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped; robot = u.scene["robot"]; cs = u.scene["feet_contact"]
        fid = robot.find_bodies(["ankle_left", "ankle_right"])[0]
        obs = env.get_observations()
        x0 = None; zs = []; contacts = []; vx = []
        for i in range(250):  # 5 s
            with torch.inference_mode():
                obs, _, d, _ = env.step(pol(obs))
            if i == 50: x0 = robot.data.root_pos_w[:, 0].clone()
            if i >= 50:
                zs.append(robot.data.body_pos_w[:, fid, 2].clone())
                contacts.append((cs.data.net_forces_w.norm(dim=-1) > 0.1).clone())
                vx.append(robot.data.root_lin_vel_b[:, 0].clone())
        x1 = robot.data.root_pos_w[:, 0]
        zs = torch.stack(zs); ct = torch.stack(contacts).float()
        print(f"[gait] cmd vx {a.vx}: mean body vx {torch.stack(vx).mean():.3f} m/s, distance in 4 s {(x1 - x0).mean():.3f} m")
        print(f"[gait] foot z min {zs.min():.4f} max {zs.max():.4f}  mean lift range per foot {(zs.max(0).values - zs.min(0).values).mean():.4f} m")
        print(f"[gait] fraction of time each foot in contact {ct.mean():.2f}; both feet down {ct.prod(-1).mean():.2f}")
        tr = (ct[1:] != ct[:-1]).float().sum(0).mean() / 2
        print(f"[gait] contact on/off cycles per foot in 4 s: {tr:.1f}; dones {int(d.sum())}")


if __name__ == "__main__":
    main()
