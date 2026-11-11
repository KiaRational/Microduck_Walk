"""Save still images of the navigation course (overview + close-ups) with the robots standing on it.

    ../../v6/run.sh scripts/snap_course.py --out_dir /tmp/snaps
"""
import argparse, os, sys
from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--task", default="Microduck-Navigation-Play")
p.add_argument("--out_dir", default="snaps")
p.add_argument("--steps", type=int, default=20)
add_launcher_args(p)
a, h = p.parse_known_args(); sys.argv = [sys.argv[0]] + h
a.enable_cameras = True  # loads Kit rendering + replicator
import gymnasium as gym, numpy as np, torch  # noqa
import microduck_lab.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config  # noqa


def main():
    env_cfg, _ = resolve_task_config(a.task, "")
    with launch_simulation(env_cfg, a):
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, UsdGeom
        import omni.usd
        env = gym.make(a.task, cfg=env_cfg); u = env.unwrapped; env.reset()
        for _ in range(a.steps):
            env.step(torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device))
        stage = omni.usd.get_context().get_stage()
        o = u.scene.env_origins[0].cpu().numpy(); r = u.scene["robot"].data.root_pos_w[0].cpu().numpy()
        views = {"top": ((0.01, 0.0, 26.0), (0, 0, 0)), "overview": ((6.5, -6.5, 5.0), (0, 0, 0)),
                 "tile": ((o[0] + 2.2, o[1] - 2.2, 1.6), (o[0], o[1], 0.0)),
                 "closeup": ((r[0] + 0.55, r[1] - 0.55, 0.32), (r[0], r[1], 0.1))}
        os.makedirs(a.out_dir, exist_ok=True)
        for name, (eye, tgt) in views.items():
            path = f"/World/snap_{name}"
            cam = UsdGeom.Camera.Define(stage, path); cam.CreateFocalLengthAttr(18.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100))
            m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse()
            UsdGeom.Xformable(cam).AddTransformOp().Set(m)
            rp = rep.create.render_product(path, (1280, 720)); an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])
            for _ in range(40):
                u.sim.render()
            img = an.get_data()
            Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(a.out_dir, f"{name}.png")); an.detach(); rp.destroy()
            print("[snap]", name, flush=True)


if __name__ == "__main__":
    main()
