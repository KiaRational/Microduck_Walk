"""Fixed-command walking test (like the reference notebook's six-direction check), with falls detected directly:
a robot has fallen if its trunk tilts more than 45 deg or drops below 7 cm at any moment, whether or not the
episode ends. For each command, N robots walk for --seconds; reports the achieved velocity and the share standing.

    ../../v6/run.sh scripts/eval_dirs.py --task Microduck-Walk-V4-Play --checkpoint <ckpt>
"""
import argparse, os, sys
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=10.0)
add_launcher_args(p)
a, h = p.parse_known_args(); sys.argv = [sys.argv[0]] + h
import gymnasium as gym, torch  # noqa
import microduck_lab.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config  # noqa

TESTS = [("stand", (0.0, 0.0, 0.0)), ("forward", (0.3, 0.0, 0.0)), ("backward", (-0.3, 0.0, 0.0)),
         ("left", (0.0, 0.2, 0.0)), ("right", (0.0, -0.2, 0.0)), ("turn_left", (0.0, 0.0, 0.8)),
         ("turn_right", (0.0, 0.0, -0.8))]


def main():
    cfg, agent_cfg = resolve_task_config(a.task, "rsl_rl_cfg_entry_point")
    if hasattr(cfg, "play_mode"):
        cfg.play_mode()
    cfg.scene.num_envs = a.num_envs
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.episode_length_s = a.seconds + 5.0
    with launch_simulation(cfg, a):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        r = create_rsl_rl_runner(env, agent_cfg); r.load(os.path.abspath(a.checkpoint))
        pol = r.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped; robot = u.scene["robot"]; cmd_term = u.command_manager.get_term("base_velocity")
        scores = []
        obs = env.get_observations()
        for name, (vx, vy, wz) in TESTS:  # tests run back to back; falls and velocity are measured per test
            fallen = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
            v_sum = torch.zeros(u.num_envs, 3, device=u.device); n = 0
            for k in range(int(a.seconds / u.step_dt)):
                with torch.inference_mode():
                    cmd_term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=u.device)
                    cmd_term.is_standing_env[:] = False
                    obs = env.get_observations()
                    obs, _, _, _ = env.step(pol(obs))
                tilt_bad = robot.data.projected_gravity_b.torch[:, 2] > -math_cos45
                low = robot.data.root_pos_w.torch[:, 2] < 0.07
                fallen |= tilt_bad | low
                if k * u.step_dt >= 2.0:  # skip the first 2 s (acceleration)
                    v = robot.data.root_lin_vel_b.torch
                    v_sum += torch.stack([v[:, 0], v[:, 1], robot.data.root_ang_vel_b.torch[:, 2]], 1); n += 1
            ok = ~fallen
            vm = (v_sum[ok] / max(n, 1)).mean(0) if ok.any() else torch.zeros(3, device=u.device)
            target = torch.tensor([vx, vy, wz], device=u.device)
            if name == "stand":
                track = 100.0 * float(torch.exp(-vm.norm() / 0.1))
            else:
                axis = int(torch.argmax(target.abs()))
                track = 100.0 * max(0.0, min(1.0, float(vm[axis] / target[axis])))
            surv = 100.0 * float(ok.float().mean())
            scores.append(surv * track / 100.0)
            print(f"[dirs] {name:10s} cmd ({vx:+.1f},{vy:+.1f},{wz:+.1f}) -> mean (vx {vm[0]:+.2f}, vy {vm[1]:+.2f}, "
                  f"wz {vm[2]:+.2f}); still standing {surv:5.1f}%; tracking {track:5.1f}%", flush=True)
        print(f"[dirs] overall score {sum(scores) / len(scores):.1f}/100 (standing% x tracking%, averaged)", flush=True)


math_cos45 = 0.7071
if __name__ == "__main__":
    main()
