"""Fall rate of a walking checkpoint, with exactly the settings ``isaaclab play`` uses (``play_mode()``): every
robot walks for ``--seconds`` with random velocity commands every 3 s; every fall is counted (robots restart after
a fall), reported as falls per robot-minute and as the share of robots that fell at least once.

    ../../v6/run.sh scripts/eval_walk.py --task Microduck-Walk-V2-Flat-Play --checkpoint <ckpt> --num_envs 256
"""
import argparse, os, sys
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--seconds", type=float, default=60.0)
add_launcher_args(p)
a, h = p.parse_known_args(); sys.argv = [sys.argv[0]] + h
import gymnasium as gym, torch  # noqa
import microduck_lab.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config  # noqa


def main():
    cfg, agent_cfg = resolve_task_config(a.task, "rsl_rl_cfg_entry_point")
    if hasattr(cfg, "play_mode"):
        cfg.play_mode()  # same as isaaclab play (and its videos)
    cfg.scene.num_envs = a.num_envs
    gen = cfg.scene.terrain.terrain_generator
    if gen is not None and getattr(cfg.events, "spread", None) is not None:
        cfg.scene.terrain.terrain_generator = gen.replace(num_cols=max(gen.num_cols, -(-a.num_envs // gen.num_rows)))
    with launch_simulation(cfg, a):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(a.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        r = create_rsl_rl_runner(env, agent_cfg); r.load(os.path.abspath(a.checkpoint))
        pol = r.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped; tm = u.termination_manager
        obs = env.get_observations()
        falls = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
        kinds = {name: 0 for name in tm.active_terms if name != "time_out"}
        for _ in range(int(a.seconds / u.step_dt)):
            with torch.inference_mode():
                obs, _, d, _ = env.step(pol(obs))
            fell = d.bool() & ~tm.time_outs
            falls += fell.long()
            for name in kinds:
                kinds[name] += int((tm.get_term(name) & fell).sum())
        n, minutes = u.num_envs, a.seconds / 60.0
        print(f"[walk-eval] {a.task}: {int(falls.sum())} falls in {n} robots x {a.seconds:.0f} s = "
              f"{falls.sum().item() / (n * minutes):.2f} falls per robot-minute; "
              f"{100 * (falls > 0).float().mean().item():.0f}% of robots fell at least once; by cause {kinds}",
              flush=True)


if __name__ == "__main__":
    main()
