"""Walker v4: flat-ground walking matched to the reference Microduck pipeline.

Reference: pollen-robotics/microduck_rl (``Mjlab-Velocity-Flat-MicroDuck``) and AMD's "MicroDuck Lab standalone"
notebook (rocm.docs.amd.com ... microduck_lab_standalone), which trains a usable walker in ~950 PPO iterations x
2048 robots. What was wrong in v1-v3 and is fixed here:

* actuator: v1-v3 used an ideal PD at 3 N*m/rad, ~5x stiffer than the real XL330 servos, with no command delay,
  no back-EMF, no current limit and no gearbox friction; the reference reward weights assume the servo model.
  v4 uses the BAM XL330 model (microduck_lab/bam.py): 15-30 ms delay, ~0.55 N*m/rad, 1.75 A limit, friction,
  battery voltage 6.5-8.2 V with sag;
* rewards: exactly the reference set and weights (tracking 2.0 with std sqrt(0.1) / sqrt(0.5), upright 2.0,
  leg pose 1.0 always on, air time 3.0 in a 0.125-0.3 s window, foot clearance -2, swing height -0.25, slip -0.1,
  joint limits -1, action rate -0.1 ramped to -1.0 by iteration 1500, body angular velocity -0.05); none of the
  v2/v3 extras (no flight penalty, no termination penalty, no trunk-height term);
* terminations: only tilt > 70 deg and time-out (20 s), as in the reference;
* pushes every 3-6 s (+-0.3 m/s), commands resampled every 3-8 s over the full range.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, math as math_utils, replace

import isaaclab_tasks.core.velocity.mdp as mdp

from microduck_lab.robot import LEG_JOINTS, MICRODUCK_BAM_CFG
from microduck_lab.tasks import gait_mdp
from microduck_lab.tasks.velocity_env_cfg import (
    _FEET_BODIES, _FEET_SENSOR, _TRUNK, MicroduckWalkV2FlatEnvCfg, MicroduckWalkV2FlatEnvCfg_PLAY,
)
from microduck_lab.tasks.walk_v3 import pose_deviation_exp

FOOT_TARGET = 0.02 + gait_mdp.SOLE_TO_ANKLE


def track_linear_velocity(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot")):
    """Reference tracking: xy velocity error in the body frame plus vertical drift."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    v = asset.data.root_link_lin_vel_b.torch
    err = torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1) + torch.square(v[:, 2])
    return gait_mdp._zero_non_finite(torch.exp(-err / std**2))


def track_angular_velocity(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot")):
    """Reference tracking: yaw-rate error plus roll/pitch rate drift."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    w = asset.data.root_link_ang_vel_b.torch
    err = torch.square(cmd[:, 2] - w[:, 2]) + torch.sum(torch.square(w[:, :2]), dim=1)
    return gait_mdp._zero_non_finite(torch.exp(-err / std**2))


def feet_air_time(env, sensor_cfg, command_name: str, threshold_min: float, threshold_max: float,
                  command_threshold: float):
    """Reference air-time reward: feet whose current swing lasts 0.125-0.3 s, while moving."""
    air = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    r = torch.sum(((air > threshold_min) & (air < threshold_max)).float(), dim=1)
    return r * (gait_mdp._command_magnitude(env, command_name) > command_threshold).float()


@configclass
class WalkV4Rewards:
    track_linear_velocity = RewTerm(func=track_linear_velocity, weight=2.0,
                                    params={"command_name": "base_velocity", "std": math.sqrt(0.1)})
    track_angular_velocity = RewTerm(func=track_angular_velocity, weight=2.0,
                                     params={"command_name": "base_velocity", "std": math.sqrt(0.5)})
    upright = RewTerm(func=gait_mdp.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK})
    pose = RewTerm(func=pose_deviation_exp, weight=1.0,
                   params={"command_name": "base_velocity", "walking_threshold": 0.01,
                           "asset_cfg": SceneEntityCfg("robot")})
    body_ang_vel = RewTerm(func=gait_mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(func=feet_air_time, weight=3.0,
                       params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold_min": 0.125,
                               "threshold_max": 0.3, "command_threshold": 0.01})
    foot_clearance = RewTerm(func=gait_mdp.foot_clearance, weight=-2.0,
                             params={"target_height": FOOT_TARGET, "command_name": "base_velocity",
                                     "asset_cfg": _FEET_BODIES, "command_threshold": 0.01})
    foot_swing_height = RewTerm(func=gait_mdp.foot_swing_height, weight=-0.25,
                                params={"sensor_cfg": _FEET_SENSOR, "asset_cfg": _FEET_BODIES,
                                        "target_height": FOOT_TARGET, "command_name": "base_velocity",
                                        "command_threshold": 0.01})
    foot_slip = RewTerm(func=gait_mdp.foot_slip, weight=-0.1,
                        params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "asset_cfg": _FEET_BODIES,
                                "command_threshold": 0.01})


def _match_reference(cfg):
    cfg.scene.robot = replace(MICRODUCK_BAM_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    cfg.sim.use_newton_actuators = False  # the servo model is a Python actuator, not a built-in one
    # Physics: Isaac Sim's PhysX everywhere. The automatic "physx" choice used the standalone ovphysx when
    # headless (training, tests) but Isaac Sim's PhysX whenever the GUI / video ran, so policies were shown on
    # a different physics engine than they were trained on. MICRODUCK_PHYSICS=ovphysx restores the old choice.
    import os
    engine = os.environ.get("MICRODUCK_PHYSICS", "isaacsim_physx")
    cfg.sim.physics.default = getattr(cfg.sim.physics, engine)
    cfg.rewards = WalkV4Rewards()
    cfg.terminations.base_too_low = None  # reference ends episodes only on > 70 deg tilt (and time-out)
    cfg.events.base_external_force_torque = None
    c = cfg.commands.base_velocity
    c.resampling_time_range = (3.0, 8.0)
    c.rel_standing_envs = 0.02
    c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0)


@configclass
class MicroduckWalkV4EnvCfg(MicroduckWalkV2FlatEnvCfg):
    ACTION_RATE_STAGES = ((0, -0.1), (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0))

    def __post_init__(self):
        super().__post_init__()
        _match_reference(self)
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class MicroduckWalkV4EnvCfg_PLAY(MicroduckWalkV2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _match_reference(self)


_ = math_utils
