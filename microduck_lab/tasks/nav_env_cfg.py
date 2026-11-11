"""Microduck goal navigation with obstacle avoidance from a head depth camera.

High level (10 Hz): observes the goal (x, y in the base frame), base velocity, gravity, its last command and a
32x24 depth image from a ray-cast camera at the head-camera position; outputs (vx, vy, wz) to the frozen walking
policy (50 Hz). Every env gets its own random obstacle-course tile (course.py);
touching an obstacle with the trunk/head ends the episode.
"""

import glob
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass, replace

import isaaclab_tasks.contrib.navigation.mdp as nav_mdp

from microduck_lab.robot import ASSET_DIR, BASE, MICRODUCK_CFG
from microduck_lab.tasks.course import MicroduckCourseCfg, full_course, goal_patches
from microduck_lab.tasks.nav_action import WalkPolicyActionCfg
from microduck_lab.tasks.velocity_env_cfg import MicroduckWalkV2EnvCfg

WALK = MicroduckWalkV2EnvCfg()  # the low-level walker's observation / action interface
NAV_USD = os.path.join(ASSET_DIR, "usd_nav", "microduck_nav_isaac.usda")
DEPTH_W, DEPTH_H, DEPTH_MAX = 32, 24, 2.0


def latest_walk_policy() -> str:
    runs = sorted(glob.glob(os.path.join(os.path.dirname(ASSET_DIR), "logs", "rsl_rl", "microduck_walk_v2", "*", "exported", "policy.pt")))
    fixed = os.path.join(ASSET_DIR, "walk_v2_policy.pt")  # final walking policy (copied from exported/policy.pt)
    return os.environ.get("MICRODUCK_WALK_POLICY", fixed if os.path.exists(fixed) else (runs[-1] if runs else ""))


def one_robot_per_tile(env, env_ids):
    """Startup event: put env i on its own terrain tile (Isaac Lab otherwise assigns tiles randomly, so several
    robots share a tile; they don't collide with each other, but they overlap and see the same obstacles)."""
    t = env.scene.terrain
    rows, cols = t.terrain_origins.shape[:2]
    idx = torch.arange(env.num_envs, device=t.env_origins.device) % (rows * cols)
    t.terrain_levels = idx % rows
    t.terrain_types = idx // rows
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]


def depth_obs(env, sensor_cfg: SceneEntityCfg, max_depth: float = DEPTH_MAX) -> torch.Tensor:
    """Depth image in [0, 1] as (N, 1, H, W) - image-shaped so RSL-RL's CNNModel encodes it with a CNN."""
    d = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"]  # (N, H, W, 1)
    d = torch.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=0.0).clamp(0.0, max_depth) / max_depth
    return d.permute(0, 3, 1, 2).contiguous()


OBSTACLE_TERRAIN = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=1.0,
    num_rows=32,
    num_cols=64,  # 2048 tiles: one per robot during training
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.3, 1.0),
    color_scheme="none",  # keep the props' vertex colours (wood, rubber, metal, cases)
    sub_terrains={
        # blog-style course: ramps / steps / platforms / mats / plates to walk over, boxes / flight cases /
        # C-stands / tape stacks to walk around (see course.py)
        "course": full_course(proportion=0.9),
        "open": MicroduckCourseCfg(proportion=0.1, num_walk_over=(0, 2), num_avoid=(0, 2), flat_patch_sampling=goal_patches()),
    },
)


@configclass
class ActionsCfg:
    walk = WalkPolicyActionCfg(
        policy_path=latest_walk_policy(),
        low_level_decimation=WALK.decimation,
        leg_actions=WALK.actions.joint_pos,
        head_actions=WALK.actions.head,
        low_level_observations=WALK.observations.policy,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        goal = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "goal"})
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        last_command = ObsTerm(func=base_mdp.last_action)

    @configclass
    class DepthCfg(ObsGroup):
        """32x24 head depth image, kept 2D for the CNN encoder."""
        image = ObsTerm(func=depth_obs, params={"sensor_cfg": SceneEntityCfg("depth_cam")})

    policy: PolicyCfg = PolicyCfg()
    depth: DepthCfg = DepthCfg()


@configclass
class CommandsCfg:
    # goals are sampled from the tile's flat patches: floor or low props, clear of tall props
    goal = base_mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot", simple_heading=True, resampling_time_range=(1e6, 1e6), debug_vis=True,
        position_success_threshold=0.15,
        ranges=base_mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-50.0)
    goal_coarse = RewTerm(func=nav_mdp.position_command_error_tanh, weight=1.0, params={"std": 1.0, "command_name": "goal"})
    goal_fine = RewTerm(func=nav_mdp.position_command_error_tanh, weight=1.0, params={"std": 0.2, "command_name": "goal"})
    command_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    hit_obstacle = DoneTerm(func=base_mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("body_contact", body_names=BASE), "threshold": 0.5})
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.8})
    base_too_low = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.07})


@configclass
class EventsCfg:
    spread = EventTerm(func=one_robot_per_tile, mode="startup")
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(func=base_mdp.reset_joints_by_scale, mode="reset",
                             params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})


@configclass
class MicroduckNavEnvCfg(ManagerBasedRLEnvCfg):
    scene = replace(WALK.scene, num_envs=2048, env_spacing=3.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    def __post_init__(self):
        self.sim = replace(WALK.sim)
        self.decimation = WALK.decimation * 5  # 10 Hz high level
        self.episode_length_s = 15.0
        self.sim.render_interval = WALK.decimation
        # robot with trunk/head colliders, obstacle terrain, sensors
        self.scene.robot = replace(MICRODUCK_CFG, prim_path="{ENV_REGEX_NS}/Robot",
                                   spawn=replace(MICRODUCK_CFG.spawn, usd_path=NAV_USD))
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator", terrain_generator=OBSTACLE_TERRAIN,
            max_init_terrain_level=None, collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=None,  # vertex colours
        )
        self.scene.left_foot_height = None  # walker-training sensors, not needed here
        self.scene.right_foot_height = None
        self.curriculum = None
        self.scene.body_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/" + BASE, history_length=3)
        self.scene.depth_cam = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/" + BASE,  # tracks the trunk; offset = head-camera pose
            spawn=None,
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.064, 0.0, 0.128), rot=(0.0, math.sin(math.radians(7.5)), 0.0,
                                                                              math.cos(math.radians(7.5))), convention="world"),
            data_types=["distance_to_image_plane"], depth_clipping_behavior="max", max_distance=DEPTH_MAX,
            pattern_cfg=PinholeCameraPatternCfg(focal_length=11.0, horizontal_aperture=20.955, width=DEPTH_W, height=DEPTH_H),
        )
        self.viewer.origin_type = "world"
        self.viewer.eye = (7.0, -7.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class MicroduckNavEnvCfg_PLAY(MicroduckNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.terrain_generator = replace(OBSTACLE_TERRAIN, num_rows=4, num_cols=4)
        self.commands.goal.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.06, 0.06, 0.15)
