from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MicroduckFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "microduck_flat"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(hidden_dims=[256, 128, 128], activation="elu", obs_normalization=False,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


# CNN encoder for the 1x24x32 depth image: 24x32 -> 10x14 -> 4x6 -> 2x4 (x32 channels = 256 features),
# concatenated with the 13 state inputs, then the MLP. Actor and critic each have their own encoder.
_DEPTH_CNN = RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 1],
                                     activation="elu")


@configclass
class MicroduckNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "microduck_nav"
    obs_groups = {"actor": ["policy", "depth"], "critic": ["policy", "depth"]}
    actor = RslRlCNNModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True, cnn_cfg=_DEPTH_CNN,
                             distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.5))
    critic = RslRlCNNModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True, cnn_cfg=_DEPTH_CNN)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


@configclass
class MicroduckWalkV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Asymmetric actor-critic like the official MicroDuck runner ([512, 256, 128], normalized inputs)."""
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "microduck_walk_v2"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
                             distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


@configclass
class MicroduckWalkV3PPORunnerCfg(MicroduckWalkV2PPORunnerCfg):
    """v2 runner + left/right mirror data augmentation (as Isaac Lab's official MicroDuck task)."""
    max_iterations = 6000
    experiment_name = "microduck_walk_v3"

    def __post_init__(self):
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        from microduck_lab.tasks.walk_v3 import mirror_states

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=False,
                                                       data_augmentation_func=mirror_states)


@configclass
class MicroduckWalkV4PPORunnerCfg(MicroduckWalkV3PPORunnerCfg):
    """Reference PPO setup: [512, 256, 128] ELU, init std 1.0, entropy 0.01, 24 steps, mirror augmentation."""
    max_iterations = 3000
    experiment_name = "microduck_walk_v4"


@configclass
class MicroduckWalkV5PPORunnerCfg(MicroduckWalkV2PPORunnerCfg):
    """Reference PPO: [512, 256, 128] ELU, init std 1.0, entropy 0.01, lr 1e-3 adaptive, mirror loss 0.5."""
    max_iterations = 3000
    experiment_name = "microduck_walk_v5"

    def __post_init__(self):
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        from microduck_lab.tasks.walk_v3 import mirror_states

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=False, use_mirror_loss=True,
                                                       mirror_loss_coeff=0.5, data_augmentation_func=mirror_states)


@configclass
class MicroduckWalkV6PPORunnerCfg(MicroduckWalkV5PPORunnerCfg):
    """v5 trained on Isaac Sim's PhysX (the engine the GUI uses)."""
    experiment_name = "microduck_walk_v6"


@configclass
class MicroduckWalkV7PPORunnerCfg(MicroduckWalkV5PPORunnerCfg):
    """v5/v6 PPO with a CNN depth encoder in front of the [512, 256, 128] actor and critic."""
    max_iterations = 4000
    experiment_name = "microduck_walk_v7"
    obs_groups = {"actor": ["policy", "depth"], "critic": ["policy", "critic", "depth"]}
    actor = RslRlCNNModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True, cnn_cfg=_DEPTH_CNN,
                             distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = RslRlCNNModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True, cnn_cfg=_DEPTH_CNN)
