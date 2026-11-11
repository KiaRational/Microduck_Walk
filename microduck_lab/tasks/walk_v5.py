"""Walker v5: Isaac Lab port of the reference MicroDuck velocity task.

Reference: rocPAI-Forge/microduck_rl_unilab ``conf/ppo/task/microduck_velocity_flat/{base,mujoco}.yaml`` (itself a
reproduction of pollen-robotics/microduck_rl at 29e887e). On top of v4 (BAM XL330 servos + reference rewards):

* the head is actuated (it is 38 % of the robot's mass) and follows a random head-pose command, rewarded by
  ``head_pose_tracking`` (weight 2, std 0.5) and ``head_pose_bias`` (weight ramped 0 -> 3); the head-pose
  command range widens in stages up to +-1.1/1.1/1.4/0.31 rad;
* actor observations as in the reference: IMU angular velocity (+-0.03 noise) and gravity (+-0.01), 0-1 step
  lag; joint positions (+-0.001); joint velocities (+-0.25) with a fixed 1-step (20 ms) delay; last action;
  velocity command; head-pose command;
* resets go exactly to the HOME pose (base xy +-0.5 m, yaw +-pi, z +0-1 cm); foot friction 0.7-1.3; trunk mass
  +-5 %; pushes every 3-6 s; share of stand-still commands grows 0.02 -> 0.25;
* PPO with the mirror loss (coefficient 0.5) instead of mirror data augmentation, as in the reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp

from microduck_lab.robot import HEAD_JOINTS
from microduck_lab.tasks import gait_mdp
from microduck_lab.tasks.walk_v4 import MicroduckWalkV4EnvCfg, MicroduckWalkV4EnvCfg_PLAY

HEAD_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
HEAD_STAGES = [  # (iteration, half-ranges for neck_pitch, head_pitch, head_yaw, head_roll) [rad]
    (0, (0.05, 0.05, 0.07, 0.015)),
    (500, (0.17, 0.17, 0.21, 0.047)),
    (1000, (0.39, 0.39, 0.49, 0.11)),
    (1500, (0.72, 0.72, 0.91, 0.2)),
    (2000, (1.1, 1.1, 1.4, 0.31)),
]


# ------------------------------------------------------------------------------------------------ head command

class HeadPoseCommand(CommandTerm):
    """Random head-joint offsets from HOME (neck_pitch, head_pitch, head_yaw, head_roll), resampled every 2-5 s."""

    cfg: HeadPoseCommandCfg

    def __init__(self, cfg: HeadPoseCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids = [self.robot.joint_names.index(n) for n in HEAD_NAMES]
        self.cmd = torch.zeros(self.num_envs, 4, device=self.device)
        self.half_range = torch.tensor(cfg.half_range, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.cmd

    def _update_metrics(self):
        q = self.robot.data.joint_pos.torch[:, self.joint_ids] - self.robot.data.default_joint_pos.torch[:, self.joint_ids]
        self.metrics["head_error"] = (q - self.cmd).abs().mean(dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.rand_like(self.cmd[env_ids]) * 2.0 - 1.0  # env_ids may be a slice
        self.cmd[env_ids] = r * self.half_range

    def _update_command(self):
        pass


@configclass
class HeadPoseCommandCfg(CommandTermCfg):
    class_type: type = HeadPoseCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    half_range: tuple[float, float, float, float] = HEAD_STAGES[0][1]


def staged_head_range(env, env_ids, command_name: str, stages):
    it = env.common_step_counter // 24
    half = [s for start, s in stages if it >= start][-1]
    term = env.command_manager.get_term(command_name)
    term.half_range[:] = torch.tensor(half, device=env.device)
    return float(half[0])


def staged_standing_envs(env, env_ids, command_name: str, stages):
    it = env.common_step_counter // 24
    frac = [s for start, s in stages if it >= start][-1]
    env.command_manager.get_term(command_name).cfg.rel_standing_envs = frac
    return frac


# ------------------------------------------------------------------------------------------------ rewards

def _head_error(env, command_name):
    term = env.command_manager.get_term(command_name)
    q = term.robot.data.joint_pos.torch[:, term.joint_ids] - term.robot.data.default_joint_pos.torch[:, term.joint_ids]
    return q - term.command


def head_pose_tracking(env, command_name: str, std: float):
    err = _head_error(env, command_name)
    return gait_mdp._zero_non_finite(torch.exp(-torch.square(err / std)).mean(dim=1))


class head_pose_bias(ManagerTermBase):
    """|1 s moving average of the head error|: penalizes a steady droop, not the oscillation of walking."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._ema = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._ema[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, command_name: str, tau_s: float):
        a = min(1.0, env.step_dt / tau_s)
        self._ema = (1.0 - a) * self._ema + a * gait_mdp._zero_non_finite(_head_error(env, command_name))
        return self._ema.abs().mean(dim=1)


def head_command_obs(env, command_name: str):
    return env.command_manager.get_command(command_name)


# ------------------------------------------------------------------------------------------------ env

def _reference_task(cfg, play: bool):
    # head is actuated (same scale as the legs)
    cfg.actions.head = replace(cfg.actions.head, scale=1.0)
    cfg.commands.head_pose = HeadPoseCommandCfg()
    # actor observations: reference noise and delays
    pol = cfg.observations.policy
    pol.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
    pol.base_ang_vel.delay_min_lag, pol.base_ang_vel.delay_max_lag = 0, 1
    pol.projected_gravity.noise = Unoise(n_min=-0.01, n_max=0.01)
    pol.projected_gravity.delay_min_lag, pol.projected_gravity.delay_max_lag = 0, 1
    pol.joint_pos.noise = Unoise(n_min=-0.001, n_max=0.001)
    pol.joint_vel.noise = Unoise(n_min=-0.25, n_max=0.25)
    pol.joint_vel.delay_min_lag, pol.joint_vel.delay_max_lag = 1, 1
    pol.head_pose_command = ObsTerm(func=head_command_obs, params={"command_name": "head_pose"})
    cfg.observations.critic.head_pose_command = ObsTerm(func=head_command_obs, params={"command_name": "head_pose"})
    # rewards
    cfg.rewards.head_pose_tracking = RewTerm(func=head_pose_tracking, weight=2.0,
                                             params={"command_name": "head_pose", "std": 0.5})
    cfg.rewards.head_pose_bias = RewTerm(func=head_pose_bias, weight=0.0,
                                         params={"command_name": "head_pose", "tau_s": 1.0})
    # events
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.01),
                                                  "yaw": (-math.pi, math.pi)}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.physics_material.params["static_friction_range"] = (0.7, 1.3)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.3)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (0.95, 1.05)
    if not play:
        cfg.curriculum.head_range = CurrTerm(func=staged_head_range,
                                             params={"command_name": "head_pose", "stages": HEAD_STAGES})
        cfg.curriculum.head_bias = CurrTerm(func=gait_mdp.staged_reward_weight,
                                            params={"term_name": "head_pose_bias",
                                                    "stages": [(0, 0.0), (600, -1.0), (1000, -2.0), (1500, -3.0)]})
        cfg.curriculum.standing = CurrTerm(func=staged_standing_envs, params={
            "command_name": "base_velocity",
            "stages": [(0, 0.02), (500, 0.05), (750, 0.1), (1000, 0.15), (1500, 0.2), (2000, 0.25)]})


@configclass
class MicroduckWalkV5EnvCfg(MicroduckWalkV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _reference_task(self, play=False)


@configclass
class MicroduckWalkV5EnvCfg_PLAY(MicroduckWalkV4EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _reference_task(self, play=True)
        self.commands.head_pose.half_range = (0.0, 0.0, 0.0, 0.0)  # head held at HOME in play / evaluation


_ = (HEAD_JOINTS, SceneEntityCfg, mdp)
