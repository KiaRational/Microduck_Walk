"""Microduck velocity-tracking (walking) task on flat ground, built on Isaac Lab's core velocity env.

Command ranges and gait timing follow pollen-robotics/microduck_rl (Mjlab-Velocity-Flat-MicroDuck):
vx +-0.4 m/s, vy +-0.3 m/s, wz +-1 rad/s, 50 Hz policy, foot air time ~0.125-0.3 s.
Everything robot-size dependent (pushes, mass DR, reset velocities, heights) is scaled down for a 0.74 kg, 25 cm robot.
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from microduck_lab.robot import BASE, FEET, HEAD_JOINTS, LEG_JOINTS, MICRODUCK_CFG

# Feet are 6 levels below trunk_base in the converted USD; sensor paths are matched one level at a time.
FEET_PRIMS = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/.*/.*/.*/ankle_.*"


@configclass
class MicroduckRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.4}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("feet_contact", body_names=FEET),
                "threshold": 0.3},
    )
    air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("feet_contact", body_names=FEET), "command_name": "base_velocity"},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("feet_contact", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw", ".*hip_roll"])},
    )
    undesired_contacts = None


@configclass
class MicroduckFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: MicroduckRewards = MicroduckRewards()

    def __post_init__(self):
        super().__post_init__()
        # physics: PhysX (the base env defaults to Newton/MuJoCo-Warp)
        self.sim.physics.default = self.sim.physics.physx
        # scene: flat plane, no height scan
        self.scene.robot = replace(MICRODUCK_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 1.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.scene.contact_forces = None  # only the soles collide; use a feet-only sensor
        self.scene.feet_contact = ContactSensorCfg(prim_path=FEET_PRIMS, history_length=3, track_air_time=True)
        # actions: legs only; the head is held at HOME (scale 0)
        self.actions.joint_pos = replace(self.actions.joint_pos, joint_names=LEG_JOINTS, scale=0.5)
        self.actions.head = replace(self.actions.joint_pos, joint_names=HEAD_JOINTS, scale=0.0)
        self.observations.policy.base_lin_vel.noise.n_min, self.observations.policy.base_lin_vel.noise.n_max = -0.05, 0.05
        self.observations.policy.joint_vel.noise.n_min, self.observations.policy.joint_vel.noise.n_max = -0.25, 0.25
        # commands (microduck_rl ranges), velocity frame commands rather than heading
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1
        cmd.debug_vis = False
        cmd.ranges.lin_vel_x = (-0.4, 0.4)
        cmd.ranges.lin_vel_y = (-0.3, 0.3)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.weight = -1.0e-3
        # events, scaled to the robot
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE
        self.events.add_base_mass.params["mass_distribution_params"] = (1 / 1.1, 1.1)
        self.events.base_com.params["asset_cfg"].body_names = BASE
        self.events.base_com.params["com_range"] = {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.003, 0.003)}
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE
        self.events.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)}
        self.events.reset_base.params["velocity_range"] = {k: (-0.05, 0.05) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        # terminations: only the soles collide, so end on a fall (tilt / low trunk) instead of base contact
        self.terminations.base_contact = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
        self.terminations.base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.07})


@configclass
class MicroduckFlatEnvCfg_PLAY(MicroduckFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (3.0, 3.0)
        # command arrows sized for a 25 cm robot (green = commanded, blue = actual velocity)
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.3)
        for vis in (self.commands.base_velocity.goal_vel_visualizer_cfg, self.commands.base_velocity.current_vel_visualizer_cfg):
            vis.markers["arrow"].scale = (0.12, 0.12, 0.12)
        # viewport over the 8x8 grid of robots (also what --video records)
        self.viewer.origin_type = "world"
        self.viewer.eye = (5.0, -5.0, 3.6)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.events.push_robot = None
        self.commands.base_velocity.debug_vis = True


# ================================================================================================ walker v2
# Gait rewards from Isaac Lab's official MicroDuck task (see gait_mdp.py), no base-velocity input (not measurable
# on the robot), trained on the walk-over part of the obstacle course (ramps <= 12 %, steps <= 2 cm, mats, plates).

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.sensors import RayCasterCfg, patterns  # noqa: E402
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg  # noqa: E402

from microduck_lab.tasks import gait_mdp  # noqa: E402
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

from microduck_lab.tasks.course import MicroduckCourseCfg, one_robot_per_tile, walk_over_course  # noqa: E402

LEFT_FOOT = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left"
RIGHT_FOOT = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right"
FOOT_HEIGHT_SENSORS = ("left_foot_height", "right_foot_height")
_FEET_SENSOR = SceneEntityCfg("feet_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True)
_FEET_BODIES = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)
_TRUNK = SceneEntityCfg("robot", body_names=[BASE])

# one 3x3 m tile per robot (4096); difficulty grows with the row index (rows = 32 difficulty levels)
WALK_TERRAIN = TerrainGeneratorCfg(
    size=(3.0, 3.0), border_width=2.0, num_rows=32, num_cols=128, horizontal_scale=0.05, vertical_scale=0.005,
    slope_threshold=0.75, use_cache=False, curriculum=True, difficulty_range=(0.1, 1.0), color_scheme="none",
    sub_terrains={
        "walk_over": walk_over_course(proportion=0.8),
        "flat": MicroduckCourseCfg(proportion=0.2, num_walk_over=(0, 0), num_avoid=(0, 0), spawn_clear_radius=0.0),
    },
)


@configclass
class WalkV2Rewards:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    track_lin_vel = RewTerm(func=gait_mdp.track_linear_velocity_heading, weight=4.0,
                            params={"command_name": "base_velocity", "std": 0.15})
    track_ang_vel = RewTerm(func=gait_mdp.track_angular_velocity_world, weight=4.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    upright = RewTerm(func=gait_mdp.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK})
    pose = RewTerm(func=gait_mdp.standing_pose, weight=1.0,
                   params={"command_name": "base_velocity", "std": 0.1, "walking_threshold": 0.01,
                           "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    body_ang_vel = RewTerm(func=gait_mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(func=gait_mdp.feet_air_time_windowed, weight=3.0,
                       params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold_min": 0.125,
                               "threshold_max": 0.3, "command_threshold": 0.01})
    foot_clearance = RewTerm(func=gait_mdp.foot_clearance, weight=-2.0,
                             params={"target_height": gait_mdp.FOOT_TARGET_HEIGHT, "command_name": "base_velocity",
                                     "asset_cfg": _FEET_BODIES, "command_threshold": 0.01,
                                     "height_sensor_names": FOOT_HEIGHT_SENSORS})
    foot_swing_height = RewTerm(func=gait_mdp.foot_swing_height, weight=-0.25 * (gait_mdp.FOOT_TARGET_HEIGHT / 0.02) ** 2,
                                params={"sensor_cfg": _FEET_SENSOR, "asset_cfg": _FEET_BODIES,
                                        "target_height": gait_mdp.FOOT_TARGET_HEIGHT, "command_name": "base_velocity",
                                        "command_threshold": 0.01, "height_sensor_names": FOOT_HEIGHT_SENSORS})
    feet_flight = RewTerm(func=gait_mdp.feet_flight, weight=-2.0,
                          params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "command_threshold": 0.01})
    foot_slip = RewTerm(func=gait_mdp.foot_slip, weight=-0.1,
                        params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "asset_cfg": _FEET_BODIES,
                                "command_threshold": 0.01})


@configclass
class WalkV2CriticCfg(ObsGroup):
    """Privileged extras for the value function only (the actor never sees these)."""
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_height = ObsTerm(func=gait_mdp.foot_height_obs, params={"asset_cfg": _FEET_BODIES, "height_sensor_names": FOOT_HEIGHT_SENSORS})
    foot_air_time = ObsTerm(func=gait_mdp.foot_air_time_obs, params={"sensor_cfg": _FEET_SENSOR})
    foot_contact = ObsTerm(func=gait_mdp.foot_contact_obs, params={"sensor_cfg": _FEET_SENSOR})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class MicroduckWalkV2EnvCfg(MicroduckFlatEnvCfg):
    ACTION_RATE_STAGES = ((0, -0.1), (3000, -0.2), (4000, -0.4), (5000, -0.6), (6000, -0.8), (7000, -1.0))
    """(iteration, action_rate_l2 weight); iterations count from the start of *this* run (also after --checkpoint)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = WalkV2Rewards()
        # actor: IMU (ang vel + gravity), command, joints, last action - no base linear velocity
        self.observations.policy.base_lin_vel = None
        self.observations.critic = WalkV2CriticCfg()
        self.actions.joint_pos = replace(self.actions.joint_pos, scale=1.0)
        # terrain: walk-over props with a difficulty curriculum
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator", terrain_generator=WALK_TERRAIN,
            max_init_terrain_level=3, collision_group=-1, visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        )
        # one robot per tile (robots never share a tile); tiles get harder row by row instead of a level curriculum
        self.curriculum.terrain_levels = None
        self.events.spread = EventTerm(func=one_robot_per_tile, mode="startup")
        # action smoothness ramp (microduck_rl: -0.1 while the gait bootstraps, tightened to -1.0 later)
        self.curriculum.action_rate = CurrTerm(
            func=gait_mdp.staged_reward_weight,
            params={"term_name": "action_rate_l2", "stages": list(self.ACTION_RATE_STAGES)})
        # end an episode at 70 deg of tilt like the official MicroDuck task (46 deg ended recoverable stumbles)
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(70.0)
        foot = RayCasterCfg(prim_path=LEFT_FOOT, mesh_prim_paths=["/World/ground"], ray_alignment="yaw",
                            pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=(0.08, 0.0)), max_distance=1.0,
                            update_period=self.sim.dt * self.decimation)
        self.scene.left_foot_height = foot
        self.scene.right_foot_height = replace(foot, prim_path=RIGHT_FOOT)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)}

    def play_mode(self):
        """Keep one robot per tile in ``isaaclab play``: the core velocity env's play_mode shrinks the terrain to
        5x5 tiles while keeping 50 robots, which put two robots on every tile."""
        from isaaclab.envs import ManagerBasedRLEnvCfg

        ManagerBasedRLEnvCfg.play_mode(self)  # caps num_envs at 50, disables observation noise
        gen = self.scene.terrain.terrain_generator
        if gen is not None and self.events.spread is not None:
            self.scene.num_envs = min(self.scene.num_envs, gen.num_rows * gen.num_cols)
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class MicroduckWalkV2EnvCfg_PLAY(MicroduckWalkV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.scene.terrain.terrain_generator = replace(WALK_TERRAIN, num_rows=8, num_cols=8)  # 64 tiles, 1 robot each
        self.commands.base_velocity.resampling_time_range = (3.0, 3.0)
        self.commands.base_velocity.debug_vis = False
        self.viewer.origin_type = "world"
        self.viewer.eye = (7.0, -7.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# ================================================================================================ simplified walker
# Stage A: flat ground (like microduck_rl's main task) until the gait is stable.
# Stage B: continue from stage A (``--checkpoint``) on gentle props (steps <= 1.5 cm, ramps <= 10 %, like the
# official Isaac Lab MicroDuck rough task) with a real terrain-level curriculum (robots move up when they walk far).

GENTLE_TERRAIN = TerrainGeneratorCfg(
    size=(3.0, 3.0), border_width=2.0, num_rows=10, num_cols=20, horizontal_scale=0.05, vertical_scale=0.005,
    slope_threshold=0.75, use_cache=False, curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none",
    sub_terrains={
        "walk_over": walk_over_course(proportion=0.75, max_riser=0.015, max_ramp=0.03, max_slope=0.10),
        "flat": MicroduckCourseCfg(proportion=0.25, num_walk_over=(0, 0), num_avoid=(0, 0), spawn_clear_radius=0.0),
    },
)


def _no_foot_rays(cfg):
    """Flat ground: foot height = height above the env origin (no ray casters needed)."""
    cfg.scene.left_foot_height = None
    cfg.scene.right_foot_height = None
    cfg.rewards.foot_clearance.params["height_sensor_names"] = ()
    cfg.rewards.foot_swing_height.params["height_sensor_names"] = ()
    cfg.observations.critic.foot_height.params["height_sensor_names"] = ()


@configclass
class MicroduckWalkV2FlatEnvCfg(MicroduckWalkV2EnvCfg):
    ACTION_RATE_STAGES = ((0, -0.1), (1500, -0.2), (2000, -0.4), (2500, -0.6), (3000, -0.8), (3500, -1.0))

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        )
        self.scene.env_spacing = 1.0
        self.events.spread = None
        _no_foot_rays(self)


@configclass
class MicroduckWalkV2FlatEnvCfg_PLAY(MicroduckWalkV2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 49
        self.scene.env_spacing = 1.5
        self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (3.0, 3.0)
        self.commands.base_velocity.debug_vis = False
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, -6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class MicroduckWalkV2GentleEnvCfg(MicroduckWalkV2EnvCfg):
    ACTION_RATE_STAGES = ((0, -1.0),)  # continues a stage-A walker that already ends at -1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GENTLE_TERRAIN
        self.scene.terrain.max_init_terrain_level = 2
        self.events.spread = None  # several robots per tile so they can move between difficulty rows
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class MicroduckWalkV2GentleEnvCfg_PLAY(MicroduckWalkV2GentleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.events.push_robot = None
        self.scene.terrain.terrain_generator = replace(GENTLE_TERRAIN, num_rows=8, num_cols=8, curriculum=False)
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.events.spread = EventTerm(func=one_robot_per_tile, mode="startup")  # one robot per tile in the viewer
        self.commands.base_velocity.resampling_time_range = (3.0, 3.0)
        self.commands.base_velocity.debug_vis = False
        self.viewer.origin_type = "world"
        self.viewer.eye = (7.0, -7.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
