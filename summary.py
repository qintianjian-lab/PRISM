import torch
from torchinfo import summary

from config.config import config
from model.modules import BuildModel

torch.random.manual_seed(42)

if __name__ == "__main__":
    device = torch.device("cpu")
    mode = "eval"

    config["blk_count"] = [blks * config["blk_ratio"] for blks in config["blk_count"]]
    config["hidden_channels"] = [
        c * config["hidden_channels_ratio"] for c in config["hidden_channels"]
    ]
    model = BuildModel(
        photo_in_channel=config["photo_in_channel"],
        photo_in_dim=config["photo_in_size"],
        mag_in_channel=config["mag_in_channel"],
        mag_in_dim=config["mag_in_size"],
        out_channel=config["out_channel"],
        blk_count=config["blk_count"],
        hidden_channels=config["hidden_channels"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        enable_spectrum=(
            config["spectrum_extractor_settings"]["enable_spectrum_auxiliary"]
            if mode == "train"
            else False
        ),
        spectrum_extractor_config=config["spectrum_extractor_settings"],
        eps=config["eps"],
    )
    model.to(device)
    summary(
        model,
        input_size=[
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
        ],
        mode=mode,
        depth=3,
        device=device,
    )
