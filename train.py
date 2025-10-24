import argparse
import os
from datetime import datetime

import lightning
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from config.config import config
from dataloader.dataloader import build_dataloader
from model.lightning_model import BuildLightningModel
from model.modules import BuildModel
from utils.tools import (
    predict_model_memory_usage,
    auto_find_memory_free_card,
    set_random_seed,
)


def train(
    model: lightning.LightningModule,
    cross_validation_fold_name: str = "fold_0",
):
    if config["run_test_when_training_end"]:
        print("[Info] Run test when training end")
    verbose = config["verbose"]
    # devices setting
    precision = config["precision"]
    predicted_memory_usage = predict_model_memory_usage(
        model=BuildModel(
            photo_in_channel=config["photo_in_channel"],
            photo_in_dim=config["photo_in_size"],
            mag_in_channel=config["mag_in_channel"],
            mag_in_dim=config["mag_in_size"],
            out_channel=config["out_channel"],
            dropout=config["dropout"],
            enable_spectrum=config["spectrum_extractor_settings"][
                "enable_spectrum_auxiliary"
            ],
            spectrum_extractor_config=config["spectrum_extractor_settings"],
            eps=config["eps"],
        ),
        input_shape=(
            [
                (
                    config["batch_size"],
                    config["photo_in_channel"],
                    config["photo_in_size"],
                    config["photo_in_size"],
                ),
                (
                    config["batch_size"],
                    config["mag_in_channel"],
                    config["mag_in_size"],
                ),
            ]
        ),
        verbose=verbose,
    )
    used_device = [
        auto_find_memory_free_card(
            config["used_device"],
            predicted_memory_usage,
            idle=True,
            idle_max_seconds=60 * 60 * 24,
            verbose=verbose,
        )
    ]
    # load dataset
    train_dataloader = build_dataloader(
        config, mode="train", cross_val_name=cross_validation_fold_name
    )
    val_dataloader = build_dataloader(
        config, mode="val", cross_val_name=cross_validation_fold_name
    )
    # log settings
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("[Info] Training start time: ", current_time)
    logger_list = []
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    tensorboard_logger = TensorBoardLogger(
        save_dir=config["log_dir"], name="{}".format(current_time)
    )
    tensorboard_logger.log_hyperparams(config)
    logger_list.append(tensorboard_logger)
    if not config["debug"] and config["enable_wandb"]:
        wandb_logger = WandbLogger(
            project=config["wandb_project_name"],
            save_dir=config["log_dir"],
            name="{}".format(current_time),
        )
        logger_list.append(wandb_logger)
    # early stopping
    early_stop_callback = EarlyStopping(
        config["monitor"],
        mode=config["mode"],
        min_delta=config["min_delta"],
        patience=config["patience"],
        verbose=True,
    )
    # make checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            config["checkpoint_dir"], "{}".format(current_time), "checkpoints"
        ),
        filename="best-{epoch}-{" + config["monitor"] + ":.5f}",
        save_top_k=1,
        monitor=config["monitor"],
        mode=config["mode"],
        save_weights_only=False,
    )
    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # init trainer
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=used_device,
        precision=precision,
        logger=logger_list,
        callbacks=(
            [checkpoint_callback, lr_monitor, early_stop_callback, RichProgressBar()]
            if config["verbose"]
            else [checkpoint_callback, lr_monitor, early_stop_callback]
        ),
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        enable_progress_bar=config["verbose"],
        check_val_every_n_epoch=1,
        fast_dev_run=config["debug"],
        enable_model_summary=config["verbose"],
        accumulate_grad_batches=1,
        gradient_clip_val=config["gradient_clip_val"],
    )
    # train
    trainer.fit(model, train_dataloader, val_dataloader)
    if config["run_test_when_training_end"]:
        # test
        print("[Info] Start test")
        best_model_path = checkpoint_callback.best_model_path
        print("[Info] best model path: ", best_model_path)
        best_model = BuildLightningModel.load_from_checkpoint(best_model_path)
        best_model.eval()
        test_dataloader = build_dataloader(
            config, mode="test", cross_val_name=cross_validation_fold_name
        )
        trainer.test(best_model, test_dataloader)


def set_model_by_config(
    random_seed: int = config["random_seed"],
) -> lightning.LightningModule:

    spectrum_extractor_settings = config["spectrum_extractor_settings"]
    return BuildLightningModel(
        random_seed=random_seed,
        eps=config["eps"],
        learn_rate=config["learn_rate"],
        cos_annealing_t_0=config["cos_annealing_t_0"],
        cos_annealing_t_mult=config["cos_annealing_t_mult"],
        cos_annealing_eta_min=config["cos_annealing_eta_min"],
        dropout=config["dropout"],
        weight_decay=config["weight_decay"],
        photo_in_channel=config["photo_in_channel"],
        photo_in_dim=config["photo_in_size"],
        mag_in_channel=config["mag_in_channel"],
        mag_in_dim=config["mag_in_size"],
        out_channel=config["out_channel"],
        enable_spectrum_auxiliary=spectrum_extractor_settings[
            "enable_spectrum_auxiliary"
        ],
        spectrum_extractor_config=spectrum_extractor_settings,
        enable_torch_2=config["enable_torch_2"],
    )


def train_with_params_search(
    debug: bool = False,
    cross_validation_fold_name: str = "fold_0",
) -> None:
    config["debug"] = debug
    if config["verbose"]:
        print("config:", config)
        set_random_seed(config["random_seed"])
        print("[Info] Random seed: ", config["random_seed"])
        model = set_model_by_config(config["random_seed"])
        train(
            model=model,
            cross_validation_fold_name=cross_validation_fold_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_validation_fold_name", type=str, default="fold_0")
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    args = parser.parse_args()

    train_with_params_search(
        debug=args.debug,
        cross_validation_fold_name=args.cross_validation_fold_name,
    )
