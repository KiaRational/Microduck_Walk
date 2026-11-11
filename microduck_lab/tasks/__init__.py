"""Gym task registrations."""

import gymnasium as gym

gym.register(
    id="Microduck-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Velocity-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Microduck-Navigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:MicroduckNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckNavPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Navigation-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:MicroduckNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckNavPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2-Gentle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2GentleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id="Microduck-Walk-V2-Gentle-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:MicroduckWalkV2GentleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV2PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
for _id, _cls in (("Microduck-Walk-V3", "MicroduckWalkV3EnvCfg"), ("Microduck-Walk-V3-Play", "MicroduckWalkV3EnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_v3:{_cls}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV3PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
for _id, _cls in (("Microduck-Walk-V4", "MicroduckWalkV4EnvCfg"), ("Microduck-Walk-V4-Play", "MicroduckWalkV4EnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_v4:{_cls}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV4PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
for _id, _cls in (("Microduck-Walk-V5", "MicroduckWalkV5EnvCfg"), ("Microduck-Walk-V5-Play", "MicroduckWalkV5EnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_v5:{_cls}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV5PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
for _id, _cls in (("Microduck-Walk-V6", "MicroduckWalkV5EnvCfg"), ("Microduck-Walk-V6-Play", "MicroduckWalkV5EnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_v5:{_cls}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV6PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
for _id, _cls in (("Microduck-Walk-V7", "MicroduckWalkV7EnvCfg"), ("Microduck-Walk-V7-Play", "MicroduckWalkV7EnvCfg_PLAY")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_v7:{_cls}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroduckWalkV7PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
