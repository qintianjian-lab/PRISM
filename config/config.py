config = {
    # -- Dev / Logging --
    "debug": False,  # pytorch lightning trainer fast_dev_run
    "wandb_project_name": "Your Wandb Project Name",
    "enable_wandb": True,
    "verbose": True,
    "run_test_when_training_end": True,
    # -- Training --
    "used_device": [0, 1, 2, 3],
    "precision": "bf16-mixed",
    "epochs": 120,
    "batch_size": 512,
    "accumulate_grad_batches": 1,
    "num_workers": 10,
    "gradient_clip_val": 20,
    # -- Dataset --
    "dataset_root_dir": "/to/your/dataset/path",
    "hdf5_file_name": "dataset.h5",
    "label_dir_name": "label",
    "photo_min_max": (
        (-9.293457984924316, 123.64601135253906),
        (-9.293457984924316, 123.64601135253906),
        (-9.293457984924316, 123.64601135253906),
    ),
    # -- Checkpoint / Early Stopping --
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    # Match sweep objective (lightning_model.on_validation_epoch_end):
    #   val_sweep_score = val_mae + 0.2 * val_outlier + max(0, val_outlier - sweep_outlier_max).
    # min_delta=1e-3: ~ separating top sweep runs (~7e-4 apart); same order as +0.001 val_mae
    # or −0.005 val_outlier holding the other fixed; looser than old val_mae min_delta (2e-3).
    "monitor": "val_sweep_score_epoch",
    "min_delta": 0.001,
    "mode": "min",
    "patience": 20,
    # -- Sweep objective (online constrained optimization) --
    # outlier definition threshold stays fixed at |delta_z| >= 0.1.
    "sweep_outlier_max": 0.065,
    # -- Optimiser --
    "learn_rate": 0.0001,
    "weight_decay": 0.01,
    "cos_annealing_t_0": 4,
    "cos_annealing_t_mult": 2,
    "cos_annealing_eta_min": 1e-12,
    "dropout": 0.2,
    # -- Architecture --
    "photo_in_channel": 3,  # DESI g r z bands
    "photo_in_size": 32,
    "mag_in_channel": 1,
    "mag_in_size": 5 + 4,  # mag + colour differences (g r z W1 W2)
    "out_channel": 5,  # number of Gaussian mixture components
    # Per-stage attention heads.
    "num_heads": [2, 4, 8, 8],
    "blk_count": [1, 1, 1, 1],
    "blk_ratio": 2,  # multiplier applied to blk_count at run-time
    "hidden_channels": [8, 16, 32, 64],
    "hidden_channels_ratio": 3,  # multiplier applied to hidden_channels at run-time
    "eps": 1e-12,
    # -- Loss Weights --
    # Total training loss:
    #   L = (1 - w_align) * NLL_estimation
    #     + w_align * cosine_alignment  (supervised layers only)
    "loss_weights": {
        "nll_loss_temperature": 2.5,  # GMM soft-max temperature for NLL
        "w_align": 0.28,  # cosine alignment loss weight
    },
    # -- Spectrum Encoder --
    "spectrum_extractor_settings": {
        "enable_spectrum_auxiliary": True,
        "spectrum_size": 7626,
        "spectrum_channel": 1,
        "spectrum_embedding_dim": 128,
        "spectrum_pretrained_ckpt_dir": "./sscnn_weight",
        "spectrum_pretrained_ckpt_name": "sscnn.ckpt",
        "spectral_supervised_layers": [2, 3],
        "spectral_layer_weights": [1.0, 1.0],
        "align_split_ratio": 0.25,
    },
}
