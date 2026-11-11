"""XL330 servo model (BAM, "Better Actuator Models", Rhoban) as an Isaac Lab explicit actuator for PhysX.

Port of the motor and friction model used by pollen-robotics/microduck_rl and Isaac Lab's official MicroDuck task
(isaaclab_newton/actuators/bam_kernels.py, upstream main 2026-11), in PyTorch. Per physics step:

* the position target is delayed by a random 3-6 physics steps (15-30 ms, re-drawn every reset);
* firmware P-control turns the position error into PWM duty: duty = (target - q) * kp_fw * error_gain,
  limited by the current limit around the back-EMF duty and by +-max_pwm;
* the DC motor gives torque = kt * V * duty / R - kt^2 * qdot / R (back-EMF damping), where the supply V is
  randomized per robot (6.5-8.2 V) and sags with the total load;
* gearbox friction (Coulomb + Stribeck + load-dependent) opposes motion; it is applied smoothly as
  -budget * tanh(qdot / v_eps) because PhysX has no per-step friction-loss constraint for explicit actuators.

With the fitted XL330 values the servo is soft: ~0.55 N*m/rad effective stiffness and ~0.64 N*m torque at the
1.75 A current limit. The previous setup used an ideal PD with 3 N*m/rad, about 5x stiffer than the real robot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass

# XL330 "m6" fit from Rhoban BAM (revision 62bd8ce), as used upstream
XL330 = dict(
    kt=0.36601349688984386, resistance=2.8113923539223227, error_gain=0.0028773775022263564, max_pwm=1.0,
    max_current=1.75, friction_base=0.004771183165566, friction_viscous=0.005359668274599504,
    friction_stribeck=0.004676345799486616, dtheta_stribeck=2.890372094130307, alpha=8.683259907618984,
    load_friction_motor=0.2667860954283698,
)


class BamServo(DelayedPDActuator):
    cfg: BamServoCfg

    def __init__(self, cfg: BamServoCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        n, j = self._num_envs, self.num_joints
        self._vin = torch.full((n, 1), cfg.vin, device=self._device)
        self._sag = torch.zeros(n, 1, device=self._device)
        self._prev_tau = torch.zeros(n, j, device=self._device)
        self._kp_scale = torch.ones(n, 1, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        k = self._vin[ids].shape[0]
        lo, hi = self.cfg.vin_range
        self._vin[ids] = torch.empty(k, 1, device=self._device).uniform_(lo, hi)
        lo, hi = self.cfg.vin_drop_gain_range
        self._sag[ids] = torch.empty(k, 1, device=self._device).uniform_(lo, hi)
        lo, hi = self.cfg.kp_scale_range
        self._kp_scale[ids] = torch.empty(k, 1, device=self._device).uniform_(lo, hi)
        self._prev_tau[ids] = 0.0

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        m = XL330
        target = self.positions_delay_buffer.compute(control_action.joint_positions)
        # supply voltage sags with the total motor load of this robot
        vin = (self._vin - self._sag * self._prev_tau.abs().sum(dim=1, keepdim=True)).clamp(min=self.cfg.vin_min)
        duty = (target - joint_pos) * (self.cfg.kp_fw * self._kp_scale) * m["error_gain"]
        center = m["kt"] * joint_vel / vin
        span = m["resistance"] * m["max_current"] / vin
        duty = torch.clamp(duty, center - span, center + span).clamp(-m["max_pwm"], m["max_pwm"])
        motor_tau = m["kt"] * vin * duty / m["resistance"] - m["kt"] ** 2 * joint_vel / m["resistance"]
        # gearbox friction: Coulomb + Stribeck + load-dependent part, plus viscous
        stribeck = torch.exp(-torch.pow(joint_vel.abs() / m["dtheta_stribeck"], m["alpha"]))
        budget = m["friction_base"] + stribeck * m["friction_stribeck"] + m["load_friction_motor"] * self._prev_tau.abs()
        friction = -budget * torch.tanh(joint_vel / self.cfg.friction_vel_eps) - m["friction_viscous"] * joint_vel
        self._prev_tau = motor_tau.clamp(-self.cfg.max_effort, self.cfg.max_effort)
        self.computed_effort = motor_tau + friction
        self.applied_effort = self.computed_effort.clamp(-self.cfg.max_effort, self.cfg.max_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class BamServoCfg(DelayedPDActuatorCfg):
    class_type: type = BamServo
    stiffness: float = 0.0  # unused: the torque comes from the motor model
    damping: float = 0.0
    kp_fw: float = 200.0
    """Firmware position P gain (microduck's setting)."""
    vin: float = 7.4
    vin_range: tuple[float, float] = (6.5, 8.2)
    vin_drop_gain_range: tuple[float, float] = (0.0, 0.2)
    vin_min: float = 6.0
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    max_effort: float = 1.0
    friction_vel_eps: float = 0.05
    """[rad/s] smoothing of the Coulomb friction sign."""
    min_delay: int = 3
    max_delay: int = 6


_ = MISSING
