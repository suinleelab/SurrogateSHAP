"""Configuration for DDPM."""


class DDPMConfig:
    """DDPM configurations."""

    cifar20_config = {
        "dataset": "cifar20",
        "image_size": 32,
        "in_ch": 3,
        "batch_size": 256,
        # Training params
        "optimizer_config": {
            "class_name": "Adam",
            "kwargs": {"lr": 1e-4},
        },
        "lr_scheduler_config": {
            "name": "constant",
            "kwargs": {"num_warmup_steps": 0},
        },
        "training_steps": {
            "retrain": 30000,
            "pruned_retrain": 30000,
            "gd": 1000,
            "lora": 1000,
        },
        "ckpt_freq": {
            "retrain": 10000,
            "pruned_retrain": 10000,
            "gd": 2000,
            "lora": 2000,
        },
        "sample_freq": {
            "retrain": 30000,
            "pruned_retrain": 30000,
            "gd": 2000,
            "lora": 2000,
        },
        "n_samples": 64,
        "unet_config": {
            "in_ch": 3,
            "mod_ch": 64,
            "out_ch": 3,
            "ch_mul": [1, 2, 2, 2],
            "num_res_blocks": 2,
            "cdim": 128,
            "use_conv": True,
            "droprate": 0.1,
            "dtype": None,
        },
    }
