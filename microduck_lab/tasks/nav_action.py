"""High-level action: (vx, vy, wz) commands executed by the frozen Microduck walking policy.

Like isaaclab_tasks.contrib.navigation's PreTrainedPolicyAction, but for the Microduck walking env, whose
actions are legs (10, scale 0.5) + head (4, scale 0 = held at HOME): the walking policy outputs 14 values and
its "actions" observation is those 14 values.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.utils import configclass, instantiate


class WalkPolicyAction(ActionTerm):
    cfg: WalkPolicyActionCfg

    def __init__(self, cfg: WalkPolicyActionCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.policy = torch.jit.load(cfg.policy_path, map_location=env.device).eval()
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._limits = torch.tensor(cfg.command_limits, device=self.device)
        self._legs: ActionTerm = instantiate(cfg.leg_actions, env)
        self._head: ActionTerm = instantiate(cfg.head_actions, env)
        self._n_legs = self._legs.action_dim
        self._ll_actions = torch.zeros(self.num_envs, self._n_legs + self._head.action_dim, device=self.device)

        def last_action(_env):
            if hasattr(env, "episode_length_buf"):
                self._ll_actions[env.episode_length_buf == 0] = 0.0
            return self._ll_actions

        obs_cfg = cfg.low_level_observations
        obs_cfg.actions.func, obs_cfg.actions.params = last_action, {}
        obs_cfg.velocity_commands.func, obs_cfg.velocity_commands.params = (lambda _env: self._cmd), {}
        self._obs = ObservationManager({"walk_policy": obs_cfg}, env)
        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._cmd[:] = torch.clamp(actions, -1.0, 1.0) * self._limits

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            with torch.inference_mode():
                out = self.policy(self._obs.compute_group("walk_policy"))
            self._ll_actions[:] = out
            self._legs.process_actions(out[:, : self._n_legs])
            self._head.process_actions(out[:, self._n_legs :])
            self._counter = 0
        self._legs.apply_actions()
        self._head.apply_actions()
        self._counter += 1


@configclass
class WalkPolicyActionCfg(ActionTermCfg):
    class_type: type = WalkPolicyAction
    asset_name: str = "robot"
    policy_path: str = MISSING
    """TorchScript walking policy (exported/policy.pt from the walking run)."""
    low_level_decimation: int = 4
    leg_actions: ActionTermCfg = MISSING
    head_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    command_limits: tuple[float, float, float] = (0.4, 0.3, 1.0)
    """Policy outputs in [-1, 1] are scaled to these (vx m/s, vy m/s, wz rad/s): the walking task's command ranges."""
