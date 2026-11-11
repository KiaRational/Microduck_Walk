"""Walker v7: v6 (BAM servos, reference recipe, Isaac Sim PhysX) + head depth camera + stairs / ramps / props.

* terrain: the walk-over part of the obstacle course (course.py): wooden stairs of 2-4 steps up and down with risers
  up to ``MAX_RISER`` (3 cm) at the hardest level, ramps up to 12 %, low platforms, studded mats, metal plates;
  10 difficulty rows x 20 columns of 3x3 m tiles, 20 % of tiles flat; Isaac Lab's terrain-level curriculum
  (robots that walk far move to harder rows, robots that fall short move down);
* the actor sees a 32x24 depth image from the head-camera position, pitched 30 deg down so the ground and steps in
  front of the feet are in view (range 1.5 m), encoded by a CNN; the critic also sees foot heights above the
  terrain (one ray per foot);
* the mirror loss reflects the depth image left/right.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass, replace

import isaaclab_tasks.core.velocity.mdp as mdp

from microduck_lab.robot import BASE
from microduck_lab.tasks.course import MicroduckCourseCfg, one_robot_per_tile, walk_over_course
from microduck_lab.tasks.velocity_env_cfg import FOOT_HEIGHT_SENSORS, LEFT_FOOT, RIGHT_FOOT
from microduck_lab.tasks.walk_v5 import MicroduckWalkV5EnvCfg, MicroduckWalkV5EnvCfg_PLAY

MAX_RISER = 0.03
DEPTH_W, DEPTH_H, DEPTH_MAX, DEPTH_PITCH_DEG = 32, 24, 1.5, 30.0

STAIRS_TERRAIN = TerrainGeneratorCfg(
    size=(3.0, 3.0), border_width=2.0, num_rows=10, num_cols=20, horizontal_scale=0.05, vertical_scale=0.005,
    slope_threshold=0.75, use_cache=False, curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none",
    sub_terrains={
        "walk_over": walk_over_course(proportion=0.8, max_riser=MAX_RISER, max_ramp=0.05, max_slope=0.12),
        "flat": MicroduckCourseCfg(proportion=0.2, num_walk_over=(0, 0), num_avoid=(0, 0), spawn_clear_radius=0.0),
    },
)


def depth_image(env, sensor_cfg: SceneEntityCfg, max_depth: float = DEPTH_MAX) -> torch.Tensor:
    """(N, 1, H, W) depth in [0, 1] for the CNN."""
    d = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"]
    d = torch.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=0.0).clamp(0.0, max_depth) / max_depth
    return d.permute(0, 3, 1, 2).contiguous()


@configclass
class DepthCfg(ObsGroup):
    image = ObsTerm(func=depth_image, params={"sensor_cfg": SceneEntityCfg("depth_cam")})


def _perception_and_terrain(cfg, play: bool):
    gen = replace(STAIRS_TERRAIN, num_rows=8, num_cols=8, curriculum=False) if play else STAIRS_TERRAIN
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=gen,
        max_init_terrain_level=None if play else 2, collision_group=-1, visual_material=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    cfg.scene.env_spacing = 3.0
    # foot height above the terrain (rewards + critic)
    foot = RayCasterCfg(prim_path=LEFT_FOOT, mesh_prim_paths=["/World/ground"], ray_alignment="yaw",
                        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=(0.08, 0.0)), max_distance=1.0,
                        update_period=cfg.sim.dt * cfg.decimation)
    cfg.scene.left_foot_height = foot
    cfg.scene.right_foot_height = replace(foot, prim_path=RIGHT_FOOT)
    for term in (cfg.rewards.foot_clearance, cfg.rewards.foot_swing_height):
        term.params["height_sensor_names"] = FOOT_HEIGHT_SENSORS
    cfg.observations.critic.foot_height.params["height_sensor_names"] = FOOT_HEIGHT_SENSORS
    # head depth camera (ray cast against the terrain mesh)
    s, c = math.sin(math.radians(DEPTH_PITCH_DEG / 2)), math.cos(math.radians(DEPTH_PITCH_DEG / 2))
    cfg.scene.depth_cam = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/" + BASE, spawn=None, mesh_prim_paths=["/World/ground"],
        update_period=cfg.sim.dt * cfg.decimation,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.064, 0.0, 0.128), rot=(0.0, s, 0.0, c), convention="world"),
        data_types=["distance_to_image_plane"], depth_clipping_behavior="max", max_distance=DEPTH_MAX,
        pattern_cfg=PinholeCameraPatternCfg(focal_length=11.0, horizontal_aperture=20.955, width=DEPTH_W,
                                            height=DEPTH_H),
    )
    cfg.observations.depth = DepthCfg()
    if play:
        cfg.events.spread = EventTerm(func=one_robot_per_tile, mode="startup")
        cfg.curriculum.terrain_levels = None
    else:
        cfg.events.spread = None
        cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class MicroduckWalkV7EnvCfg(MicroduckWalkV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _perception_and_terrain(self, play=False)


@configclass
class MicroduckWalkV7EnvCfg_PLAY(MicroduckWalkV5EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _perception_and_terrain(self, play=True)
        self.scene.num_envs = 64
        self.viewer.eye = (7.0, -7.0, 5.0)
