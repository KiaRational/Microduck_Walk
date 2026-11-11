"""Walker v3: flat-ground walking only, with the reward set rebuilt after the v2 analysis.

v2 (flat stage) walked but: tracked turning poorly (0.58 rad/s error on +-1 rad/s), pushed joints into their limits
(joint-limit penalty grew 5x), crouched until the trunk dropped below 7 cm (5 % of episodes) and had no posture
term while walking (the pose reward was standing-only). v3 changes, following Isaac Lab's official MicroDuck task
(isaaclab_tasks/contrib/microduck) and pollen-robotics/microduck_rl:

* posture: per-joint Gaussian pose reward that is always on, tight when standing and looser when walking
  (upstream ``joint_deviation_exp`` std values), plus a trunk-height floor penalty;
* joint limits penalized 2x harder;
* turning tracked with upstream's flat std (0.35 rad/s);
* command curriculum: small speeds first, full range (+-0.4, +-0.3 m/s, +-1 rad/s) from iteration 2000;
* left/right mirror data augmentation (upstream symmetry layout, resolved by joint name);
* no "trunk below 7 cm" termination (crouching is penalized instead); falls end at 70 deg of tilt or 5 cm height.
"""

from __future__ import annotations

import math
import re

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from microduck_lab.robot import BASE, LEG_JOINTS
from microduck_lab.tasks import gait_mdp
from microduck_lab.tasks.velocity_env_cfg import (
    _FEET_BODIES, _FEET_SENSOR, _TRUNK, MicroduckWalkV2FlatEnvCfg, MicroduckWalkV2FlatEnvCfg_PLAY,
)

FOOT_TARGET_FLAT = 0.02 + gait_mdp.SOLE_TO_ANKLE  # 2 cm swing clearance (upstream value)

# upstream per-joint pose tolerances [rad]
STD_STANDING = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle": 0.1}
STD_WALKING = {"hip_yaw": 0.3, "hip_roll": 0.05, "hip_pitch": 0.4, "knee": 0.4, "ankle": 0.25}


# ------------------------------------------------------------------------------------------------ rewards

class pose_deviation_exp(ManagerTermBase):
    """exp(-mean((q - q_default)^2 / std^2)) over the leg joints; std is per joint, tighter when standing."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        names = env.scene[asset_cfg.name].joint_names
        ids = [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in LEG_JOINTS)]
        self._ids = torch.tensor(ids, device=env.device)

        def stds(table):
            return torch.tensor([next(v for k, v in table.items() if names[i].endswith(k)) for i in ids],
                                device=env.device).unsqueeze(0)

        self._std_stand, self._std_walk = stds(STD_STANDING), stds(STD_WALKING)

    def __call__(self, env, command_name: str, walking_threshold: float, asset_cfg: SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
        standing = (gait_mdp._command_magnitude(env, command_name) < walking_threshold).unsqueeze(1)
        std = torch.where(standing, self._std_stand, self._std_walk)
        err = torch.square(asset.data.joint_pos.torch[:, self._ids] - asset.data.default_joint_pos.torch[:, self._ids])
        return gait_mdp._zero_non_finite(torch.exp(-torch.mean(err / std**2, dim=1)))


def trunk_height_floor(env, min_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """How far the trunk is below ``min_height`` [m] (0 above it): discourages the deep crouch."""
    z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return gait_mdp._zero_non_finite((min_height - z).clamp(min=0.0))


@configclass
class WalkV3Rewards:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    track_lin_vel = RewTerm(func=gait_mdp.track_linear_velocity_heading, weight=4.0,
                            params={"command_name": "base_velocity", "std": 0.15})
    track_ang_vel = RewTerm(func=gait_mdp.track_angular_velocity_world, weight=4.0,
                            params={"command_name": "base_velocity", "std": 0.35})
    upright = RewTerm(func=gait_mdp.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK})
    pose = RewTerm(func=pose_deviation_exp, weight=1.0,
                   params={"command_name": "base_velocity", "walking_threshold": 0.01,
                           "asset_cfg": SceneEntityCfg("robot")})
    trunk_height = RewTerm(func=trunk_height_floor, weight=-20.0, params={"min_height": 0.10})
    body_ang_vel = RewTerm(func=gait_mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(func=gait_mdp.feet_air_time_windowed, weight=3.0,
                       params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold_min": 0.125,
                               "threshold_max": 0.3, "command_threshold": 0.01})
    feet_flight = RewTerm(func=gait_mdp.feet_flight, weight=-1.0,
                          params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "command_threshold": 0.01})
    foot_clearance = RewTerm(func=gait_mdp.foot_clearance, weight=-2.0,
                             params={"target_height": FOOT_TARGET_FLAT, "command_name": "base_velocity",
                                     "asset_cfg": _FEET_BODIES, "command_threshold": 0.01})
    foot_swing_height = RewTerm(func=gait_mdp.foot_swing_height, weight=-0.25 * (FOOT_TARGET_FLAT / 0.02) ** 2,
                                params={"sensor_cfg": _FEET_SENSOR, "asset_cfg": _FEET_BODIES,
                                        "target_height": FOOT_TARGET_FLAT, "command_name": "base_velocity",
                                        "command_threshold": 0.01})
    foot_slip = RewTerm(func=gait_mdp.foot_slip, weight=-0.1,
                        params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "asset_cfg": _FEET_BODIES,
                                "command_threshold": 0.01})


# ------------------------------------------------------------------------------------------------ curriculum

COMMAND_STAGES = [  # (iteration, (vx, vy, wz) half-ranges)
    (0, (0.2, 0.15, 0.5)),
    (1000, (0.3, 0.22, 0.75)),
    (2000, (0.4, 0.3, 1.0)),
]


def staged_command_ranges(env, env_ids, command_name: str, stages):
    """Widen the velocity command ranges in stages (iterations counted from the start of this run)."""
    it = env.common_step_counter // 24
    vx, vy, wz = [s for start, s in stages if it >= start][-1]
    r = env.command_manager.get_term(command_name).cfg.ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-vx, vx), (-vy, vy), (-wz, wz)
    return vx


# ------------------------------------------------------------------------------------------------ symmetry

_SIGNED = {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": -1, "knee": -1, "ankle": -1,
           "neck_pitch": 1, "head_pitch": 1, "head_yaw": -1, "head_roll": -1}
_TERM_SIGNS = {"base_ang_vel": (-1, 1, -1), "projected_gravity": (1, -1, 1), "velocity_commands": (1, -1, -1),
               "base_lin_vel": (1, -1, 1), "head_pose_command": (1, 1, -1, -1)}
_cache: dict = {}


def _mirror_joints(names: list[str]) -> tuple[list[int], list[int]]:
    """Permutation and signs that reflect a list of joints left <-> right (upstream MicroDuck sign table)."""
    perm, signs = [], []
    for n in names:
        other = n.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
        perm.append(names.index(other))
        signs.append(next(s for k, s in _SIGNED.items() if n.endswith(k)))
    return perm, signs


def _layout(env):
    if "groups" in _cache:
        return _cache
    u = env.unwrapped
    joint_perm, joint_signs = _mirror_joints(list(u.scene["robot"].joint_names))
    act_names = []
    for term in u.action_manager._terms.values():
        act_names += list(term._joint_names)
    act_perm, act_signs = _mirror_joints(act_names)
    groups = {}
    for g in ("policy", "critic"):
        perm, signs = [], []
        for name, (dim,) in zip(u.observation_manager.active_terms[g], u.observation_manager.group_obs_term_dim[g]):
            if name in ("joint_pos", "joint_vel"):
                p, s = joint_perm, joint_signs
            elif name == "actions":
                p, s = act_perm, act_signs
            elif name in ("foot_height", "foot_air_time", "foot_contact"):
                p, s = [1, 0], [1, 1]
            elif name in _TERM_SIGNS:
                s = list(_TERM_SIGNS[name]); p = list(range(len(s)))
            else:
                raise ValueError(f"no mirror rule for observation term {name!r}")
            assert len(p) == dim, (name, dim)
            start = len(perm)
            perm += [start + i for i in p]
            signs += s
        groups[g] = (torch.tensor(perm, device=u.device), torch.tensor(signs, device=u.device, dtype=torch.float32))
    _cache.update(groups=groups, act=(torch.tensor(act_perm, device=u.device),
                                      torch.tensor(act_signs, device=u.device, dtype=torch.float32)))
    return _cache


@torch.no_grad()
def mirror_states(env=None, obs=None, actions=None):
    """RSL-RL data augmentation: append left/right mirrored copies of observations and actions."""
    lay = _layout(env)
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        n = obs.batch_size[0]
        for g, values in obs.items():
            if g == "depth":  # (N, 1, H, W) image: a left/right reflection flips the columns
                obs_aug[g][n:] = torch.flip(values, dims=[-1])
                continue
            perm, signs = lay["groups"][g]
            obs_aug[g][n:] = values[..., perm] * signs
    act_aug = None
    if actions is not None:
        perm, signs = lay["act"]
        act_aug = torch.cat((actions, actions[..., perm] * signs), dim=0)
    return obs_aug, act_aug


# ------------------------------------------------------------------------------------------------ env

@configclass
class MicroduckWalkV3EnvCfg(MicroduckWalkV2FlatEnvCfg):
    ACTION_RATE_STAGES = ((0, -0.1), (2000, -0.2), (3000, -0.4), (4000, -0.6), (5000, -0.8), (6000, -1.0))

    def __post_init__(self):
        super().__post_init__()
        self.rewards = WalkV3Rewards()
        self.curriculum.commands = CurrTerm(func=staged_command_ranges,
                                            params={"command_name": "base_velocity", "stages": COMMAND_STAGES})
        vx, vy, wz = COMMAND_STAGES[0][1]
        c = self.commands.base_velocity.ranges
        c.lin_vel_x, c.lin_vel_y, c.ang_vel_z = (-vx, vx), (-vy, vy), (-wz, wz)
        self.terminations.base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})


@configclass
class MicroduckWalkV3EnvCfg_PLAY(MicroduckWalkV2FlatEnvCfg_PLAY):
    """Play / evaluation: full command range, v3 rewards (only matter for logging)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = WalkV3Rewards()
        self.terminations.base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})
        c = self.commands.base_velocity.ranges
        c.lin_vel_x, c.lin_vel_y, c.ang_vel_z = (-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0)


_ = BASE
