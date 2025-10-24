config = {
    # dev mode
    "debug": False,  # pytorch lightning trainer fast_dev_run
    "wandb_project_name": "Your Wandb Project Name",
    "enable_wandb": True,  # enable wandb
    "verbose": True,
    "run_test_when_training_end": True,
    # training settings
    "random_seed": 42,
    "used_device": [0],
    "precision": "32-true",
    "dataset_root_dir": "/path/to/your/dataset",
    # dataset settings
    "photometric_dir_name": "photo",
    "label_dir_name": "label",
    # spectra settings
    "spectrum_extractor_settings": {
        "enable_spectrum_auxiliary": True,
        "spectrum_dir_name": "spec",
        "spectrum_size": 7626,
        "spectrum_channel": 1,
    },
    # torch 2.0
    "enable_torch_2": False,
    # parameters
    "photo_in_channel": 6,  # DESI g r i z， WISE W1 W2
    "photo_in_size": 32,
    "photo_min_max": (
        (-1.4950683, 29.80774),
        (-1.4950683, 29.80774),
        (-1.4950683, 29.80774),
        (-1.4950683, 29.80774),
        (3.493501, 714.04065),
        (3.493501, 714.04065),
    ),
    "mag_in_channel": 1,
    "mag_in_size": 5 + 4,  # mag + mag diff (g, r, z, W1, W2)
    "out_channel": 5,
    # others
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "monitor": "val_mae_epoch",
    "min_delta": 0.002,
    "mode": "min",
    "eps": 1e-12,
    "patience": 150,
    "gradient_clip_val": 20,
    # model settings
    "batch_size": 32,
    "num_workers": 32,
    "epochs": 1000,
    "learn_rate": 0.0001,
    "cos_annealing_t_0": 40,
    "cos_annealing_t_mult": 3,
    "cos_annealing_eta_min": 1e-12,
    "dropout": 0.1,
    "weight_decay": 0.001,
}
