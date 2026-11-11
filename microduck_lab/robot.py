"""Microduck articulation config.

The model comes from pollen-robotics/microduck_rl (robot_walk.xml; MJCF Apache-2.0, meshes CC BY-NC-SA 4.0,
(c) Pollen Robotics), cleaned (no self-collision-only geoms / sensors) and converted with Isaac Lab's MJCF
converter. Servos are simple PD (ImplicitActuator); the official stack uses the BAM XL330 model instead.
"""

import os

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
USD_PATH = os.path.join(ASSET_DIR, "usd", "microduck_walk", "microduck_walk.usda")

# HOME / STAND2 pose from microduck_rl (robot/microduck_constants.py).
HOME_JOINT_POS = {
    ".*hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "right_hip_pitch": 0.4579,
    "left_knee": -0.0049,
    "right_knee": 0.0049,
    "left_ankle": 0.4530,
    "right_ankle": -0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}

LEG_JOINTS = [".*hip_yaw", ".*hip_roll", ".*hip_pitch", ".*knee", ".*ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
FEET = ["ankle_left", "ankle_right"]
BASE = "trunk_base"

STIFFNESS = 3.0  # N*m/rad
DAMPING = 0.08  # N*m*s/rad
EFFORT = 0.96  # N*m, XL330 force range in the MJCF
ARMATURE = 0.0018  # kg*m^2, from the MJCF "chosen_actuator" class

MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=[PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=8,
                                                 solver_velocity_iteration_count=4)],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.125), joint_pos=HOME_JOINT_POS, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=STIFFNESS,
            damping=DAMPING,
            effort_limit_sim=EFFORT,
            velocity_limit_sim=8.0,
            armature=ARMATURE,
        ),
    },
)


# Same robot with the XL330 servo model (BAM) instead of the stiff ideal PD above (see bam.py).
def _bam_cfg():
    from isaaclab.utils import replace

    from microduck_lab.bam import BamServoCfg

    return replace(MICRODUCK_CFG, actuators={
        "servos": BamServoCfg(joint_names_expr=[".*"], armature=ARMATURE, effort_limit_sim=1.0, velocity_limit_sim=20.0),
    })


MICRODUCK_BAM_CFG = _bam_cfg()
