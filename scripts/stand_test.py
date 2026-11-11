"""Drop several Microducks in the HOME pose with different PD stiffness and report height / tilt."""
import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--stiffness", default="0.55,1.5,3.0,6.0")
parser.add_argument("--seconds", type=float, default=4.0)
add_launcher_args(parser)
args_cli = parser.parse_args()

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from microduck_lab.robot import MICRODUCK_CFG  # noqa: E402

GAINS = [float(v) for v in args_cli.stiffness.split(",")]


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = SimulationCfg(dt=0.005)
    with launch_simulation(sim_cfg, args_cli) as _:
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view((0.6, -0.6, 0.35), (0.0, 0.0, 0.08))
        scene = InteractiveScene(SceneCfg(num_envs=len(GAINS), env_spacing=0.5))
        sim.reset()
        robot = scene["robot"]
        print("[stand] joints:", robot.joint_names)
        print("[stand] bodies:", robot.body_names, "total mass", round(robot.data.default_mass.sum(-1)[0].item(), 3), "kg")
        k = torch.tensor(GAINS, device=sim.device)[:, None].expand(-1, robot.num_joints).contiguous()
        robot.write_joint_stiffness_to_sim(k)
        target = robot.data.default_joint_pos.clone()
        steps = int(args_cli.seconds / sim_cfg.dt)
        for i in range(steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            if i % int(0.5 / sim_cfg.dt) == 0 or i == steps - 1:
                z = (robot.data.root_pos_w[:, 2]).tolist()
                g = robot.data.projected_gravity_b[:, 2].tolist()  # -1 = upright
                err = (robot.data.joint_pos - target).abs().max(-1).values.tolist()
                print(f"[stand] t={i * sim_cfg.dt:4.1f}s " + " | ".join(
                    f"kp {GAINS[e]:.2f}: z {z[e]:.3f} up {-g[e]:.2f} jerr {err[e]:.2f}" for e in range(len(GAINS))), flush=True)


if __name__ == "__main__":
    main()
