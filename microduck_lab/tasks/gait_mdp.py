"""Gait rewards and observations for walker v2, ported from Isaac Lab's official MicroDuck task
(isaaclab_tasks/contrib/microduck, upstream main 2026-11; BSD-3-Clause) to this PhysX setup.

The v1 walker only had a weak air-time term and learned a 7 Hz foot shuffle with 6 mm lift; these terms
reward 0.125-0.3 s swing phases and a ~3 cm foot clearance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import body_projected_gravity_b
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

SOLE_TO_ANKLE = 0.0225
"""Standing ankle-frame height above the sole [m]."""
FOOT_TARGET_HEIGHT = 0.03 + SOLE_TO_ANKLE
"""Ankle height for a 3 cm swing clearance [m] (upstream uses 2 cm; our stairs have risers up to 3.5 cm)."""


def _zero_non_finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _command_magnitude(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


def foot_height(env, asset_cfg: SceneEntityCfg, height_sensor_names: tuple[str, ...] = ()) -> torch.Tensor:
    """Ankle-frame height above the terrain directly below each foot [m] (one ray caster per foot)."""
    asset = env.scene[asset_cfg.name]
    heights = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    flat = heights - env.scene.env_origins[:, 2].unsqueeze(-1)
    if not height_sensor_names:
        return flat
    out = []
    for i, name in enumerate(height_sensor_names):
        sensor = env.scene[name]
        hit_z = sensor.data.ray_hits_w.torch[..., 2]
        valid = torch.isfinite(hit_z)
        clearance = (heights[:, i, None] - hit_z).clamp(min=0.0)
        clearance = torch.where(valid, clearance, sensor.cfg.max_distance).amin(dim=-1)
        out.append(torch.where(valid.any(dim=-1), clearance, flat[:, i].clamp(0.0, sensor.cfg.max_distance)))
    return torch.stack(out, dim=-1)


def foot_height_obs(env, asset_cfg: SceneEntityCfg, height_sensor_names: tuple[str, ...] = ()) -> torch.Tensor:
    return _zero_non_finite(foot_height(env, asset_cfg, height_sensor_names))


def foot_air_time_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]


def foot_contact_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return (env.scene.sensors[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()


# ------------------------------------------------------------------------------------------------ rewards

def upright(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    g = body_projected_gravity_b(env, asset_cfg)
    xy = torch.nan_to_num(torch.sum(torch.square(g[:, :2]), dim=1), nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy / std**2)


def track_linear_velocity_heading(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_link_quat_w.torch),
                                           asset.data.root_link_lin_vel_w.torch)
    return _zero_non_finite(torch.exp(-torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1) / std**2))


def track_angular_velocity_world(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    err = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w.torch[:, 2])
    return _zero_non_finite(torch.exp(-err / std**2))


def standing_pose(env, command_name: str, std: float, walking_threshold: float, asset_cfg: SceneEntityCfg):
    """Stay near the default leg pose, only while the command is (near) zero."""
    asset = env.scene[asset_cfg.name]
    standing = _command_magnitude(env, command_name) < walking_threshold
    err = torch.square(asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids])
    return _zero_non_finite(torch.exp(-torch.mean(err, dim=1) / std**2)) * standing.float()


def feet_air_time_windowed(env, sensor_cfg: SceneEntityCfg, command_name: str, threshold_min: float,
                           threshold_max: float, command_threshold: float) -> torch.Tensor:
    """Reward each foot while its current swing has lasted between threshold_min and threshold_max [s]."""
    sensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    single_stance = (in_contact.sum(dim=1) == 1).float()  # walking, not hopping: the other foot is down
    reward = torch.sum(((air > threshold_min) & (air < threshold_max)).float(), dim=1) * single_stance
    return reward * (_command_magnitude(env, command_name) > command_threshold).float()


def feet_flight(env, sensor_cfg: SceneEntityCfg, command_name: str, command_threshold: float) -> torch.Tensor:
    """1 while both feet are off the ground and the robot is commanded to move (penalize hopping / running)."""
    in_contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    return (~in_contact.any(dim=1)).float() * (_command_magnitude(env, command_name) > command_threshold).float()


def foot_clearance(env, target_height: float, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float,
                   height_sensor_names: tuple[str, ...] = ()) -> torch.Tensor:
    """Penalize moving feet whose height differs from target_height (weighted by foot speed)."""
    asset = env.scene[asset_cfg.name]
    h = foot_height(env, asset_cfg, height_sensor_names)
    v = torch.norm(asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    cost = torch.sum(torch.abs(h - target_height) * v, dim=1)
    return _zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


class foot_swing_height(ManagerTermBase):
    """Penalize, at touchdown, how far the swing's peak height was from target_height."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._peak = torch.zeros(env.num_envs, len(cfg.params["sensor_cfg"].body_names), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._peak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, sensor_cfg, asset_cfg, target_height, command_name, command_threshold,
                 height_sensor_names: tuple[str, ...] = ()):
        sensor = env.scene.sensors[sensor_cfg.name]
        in_air = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        h = _zero_non_finite(foot_height(env, asset_cfg, height_sensor_names))
        self._peak = torch.where(in_air, torch.maximum(self._peak, h), self._peak)
        first = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        cost = torch.sum(torch.square(self._peak / target_height - 1.0) * first.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()
        self._peak = torch.where(first, torch.zeros_like(self._peak), self._peak)
        return _zero_non_finite(cost)


def foot_slip(env, sensor_cfg: SceneEntityCfg, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float):
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact = (sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    v = torch.norm(asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    cost = torch.sum(torch.square(v) * contact, dim=1)
    return _zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


def body_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return _zero_non_finite(torch.sum(torch.square(w), dim=(1, 2)))


def staged_reward_weight(env, env_ids, term_name: str, stages: list[tuple[int, float]]):
    """Curriculum: set reward ``term_name``'s weight from (iteration, weight) stages (24 env steps per iteration),
    like microduck_rl's action_rate ramp (gentle while the gait forms, smoother later)."""
    it = env.common_step_counter // 24
    weight = None
    for start, w in stages:
        if it >= start:
            weight = w
    if weight is not None:
        cfg = env.reward_manager.get_term_cfg(term_name)
        if cfg.weight != weight:
            cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, cfg)
    return weight if weight is not None else 0.0
