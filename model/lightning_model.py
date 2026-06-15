import bitsandbytes as bnb
import lightning
import torch

from model.loss import (
    NLLLoss,
    delta_z,
    nmad_z,
    z_bias,
    sigma_n,
    outlier_fraction,
    z_estimate,
    CosineSimilarityLoss,
)
from model.modules import BuildModel
from model.sscnn import SSCNNEncoder


class BuildLightningModel(lightning.LightningModule):

    def __init__(
        self,
        random_seed: int,
        eps: float | int,
        # hyper params
        learn_rate: float,
        weight_decay: float,
        cos_annealing_t_0: int,
        cos_annealing_t_mult: int,
        cos_annealing_eta_min: float,
        dropout: float,
        # photometric & mags settings
        photo_in_channel: int,
        photo_in_dim: int,
        mag_in_channel: int,
        mag_in_dim: int,
        blk_count: list[int],
        hidden_channels: list[int],
        num_heads: list[int],
        out_channel: int,
        # spectrum settings
        spectrum_extractor_config: dict,
        enable_spectrum_auxiliary: bool,
        # loss weights (all weights live here; nothing is hardcoded below)
        loss_config: dict,
        # sweep objective settings
        sweep_outlier_max: float,
    ):
        super().__init__()
        print("[INFO] Using Random Seed: ", random_seed)
        self.model = BuildModel(
            photo_in_channel=photo_in_channel,
            photo_in_dim=photo_in_dim,
            mag_in_channel=mag_in_channel,
            mag_in_dim=mag_in_dim,
            out_channel=out_channel,
            blk_count=blk_count,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            enable_spectrum=enable_spectrum_auxiliary,
            spectrum_extractor_config=spectrum_extractor_config,
            eps=eps,
        )

        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min

        self.estimation_loss_temperature = loss_config["nll_loss_temperature"]
        self.estimation_loss = NLLLoss(
            temperature=self.estimation_loss_temperature, eps=eps
        )
        self.w_align = loss_config["w_align"]
        self.contrastive_loss = CosineSimilarityLoss(
            eps=eps,
        )

        self.enable_spectrum_auxiliary = enable_spectrum_auxiliary
        self.sweep_outlier_max = float(sweep_outlier_max)
        if enable_spectrum_auxiliary:
            _embedding_dim = spectrum_extractor_config["spectrum_embedding_dim"]
            ckpt_path = spectrum_extractor_config["spectrum_pretrained_ckpt_path"]
            self.spectrum_encoder = SSCNNEncoder(
                in_channel=spectrum_extractor_config["spectrum_channel"],
                spectrum_size=spectrum_extractor_config["spectrum_size"],
                embedding_dim=_embedding_dim,
                pretrained_ckpt_path=ckpt_path,
            )
            assert (
                ckpt_path != ""
            ), "spectrum_pretrained_ckpt_path must be provided when enable_spectrum_auxiliary is True!"
            self.spectrum_encoder.requires_grad_(False)
            self.spectrum_encoder.eval()
            print("[INFO] Spectrum encoder is loaded and frozen for alignment.")
            self.spectral_supervised_layers = spectrum_extractor_config[
                "spectral_supervised_layers"
            ]
            self.spectral_layer_weights = spectrum_extractor_config[
                "spectral_layer_weights"
            ]
            print(
                "[INFO] w_align={}, nll_temp={}, supervised_layers={}".format(
                    self.w_align,
                    loss_config["nll_loss_temperature"],
                    self.spectral_supervised_layers,
                )
            )

        self.val_loss = torch.zeros(1).to(self.device)
        self.val_label = []
        self.val_pred = []
        self.val_loss_epoch = 0.0
        self.best_val_loss = 9e10
        self.best_val_mae_epoch = 9e10
        self.best_val_outlier_fraction_epoch = 1.0
        self.best_val_sweep_score_epoch = 9e10

        self.test_label = []
        self.test_pred = []

        self.step = 0
        self.eps = eps

        self.save_hyperparameters()

    def _log_eval_metrics(
        self,
        prefix: str,
        label: list[torch.Tensor] | torch.Tensor,
        pred: list[torch.Tensor] | torch.Tensor,
    ) -> dict:
        _res_dict = {}
        is_step = isinstance(label, torch.Tensor)
        suffix = "step" if is_step else "epoch"
        if isinstance(pred, list):
            pred = torch.cat(pred, dim=0)
        if isinstance(label, list):
            label = torch.cat(label, dim=0)
        _mae_pred = z_estimate(
            pred=pred,
            mode="mean",
        )
        _mae = torch.mean((_mae_pred - label.squeeze(-1)).abs())
        self.log(
            f"{prefix}_mae_{suffix}",
            _mae,
            prog_bar=True,
            on_step=is_step,
            on_epoch=not is_step,
        )
        _res_dict[f"{prefix}_mae_{suffix}"] = _mae.item()
        _d_z = delta_z(
            pred=pred,
            label=label,
            mode="mean",
        )
        metrics = [
            ("nmad_z", nmad_z(_d_z, factor=1.4826)),
            ("z_bias", z_bias(_d_z)),
            ("sigma_0.05", sigma_n(_d_z, n=0.05)),
            ("sigma_0.15", sigma_n(_d_z, n=0.15)),
            ("outlier_fraction", outlier_fraction(_d_z, threshold=0.1)),
        ]
        for metric_name, metric_value in metrics:
            key = f"{prefix}_{metric_name}_{suffix}"
            self.log(
                key,
                metric_value,
                prog_bar=True,
                on_step=is_step,
                on_epoch=not is_step,
            )
            _res_dict[key] = metric_value
        return _res_dict

    def _get_estimation_loss(
        self,
        label: torch.Tensor,
        pred: torch.Tensor,
        mode: str = "train",
    ):
        if mode not in ["train", "val", "test"]:
            raise NotImplementedError(
                f"Mode {mode} not implemented, must be in ['train', 'val', 'test']"
            )
        _estimation_loss, _weights = self.estimation_loss(pred, label)
        self.log(
            f"{mode}_estimation_loss",
            _estimation_loss,
            prog_bar=True,
            on_step=True,
        )
        return _estimation_loss

    def training_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            flux,
            south_north_flag,
        ) = batch
        if self.enable_spectrum_auxiliary:
            pred, spectrum_features, _ = self.model(
                photometric=photometric,
                magnitudes=mags,
                output_spectrum=True,
            )
            spec_emb, _ = self.spectrum_encoder(flux)

            _align_loss = torch.tensor(0.0, device=self.device)
            for local_idx, layer_idx in enumerate(self.spectral_supervised_layers):
                w = self.spectral_layer_weights[local_idx]
                photo_emb = spectrum_features[layer_idx]
                _layer_align_loss = self.contrastive_loss(
                    photo_emb,
                    spec_emb.detach(),
                )
                self.log(
                    f"train_deep_contrastive_loss_{layer_idx + 1}",
                    _layer_align_loss,
                    prog_bar=True,
                    on_step=True,
                )
                _align_loss = _align_loss + w * _layer_align_loss
            self.log(
                "train_contrastive_loss_avg", _align_loss, prog_bar=True, on_step=True
            )
        else:
            pred, _, _ = self.model(
                photometric=photometric,
                magnitudes=mags,
                output_spectrum=False,
            )
        _estimation_loss = self._get_estimation_loss(
            label=label,
            pred=pred,
            mode="train",
        )
        self.log(
            "train_estimation_loss",
            _estimation_loss,
            prog_bar=True,
            on_step=True,
        )
        self._log_eval_metrics(
            "train",
            label=label.detach().cpu(),
            pred=pred.detach().cpu(),
        )
        if self.enable_spectrum_auxiliary:
            return (1 - self.w_align) * _estimation_loss + self.w_align * _align_loss
        else:
            return _estimation_loss

    def on_validation_epoch_start(self):
        self.val_loss = torch.zeros(1).to(self.device)
        self.val_loss_epoch = 0
        self.step = 0
        self.val_label = []
        self.val_pred = []

    def validation_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            flux,
            south_north_flag,
        ) = batch
        pred, _, _ = self.model(
            photometric=photometric,
            magnitudes=mags,
            output_spectrum=False,
        )
        _estimation_loss = self._get_estimation_loss(
            label=label,
            pred=pred,
            mode="val",
        )
        self.val_loss_epoch += _estimation_loss
        self.step += 1
        self.val_label.append(label.detach().cpu())
        self.val_pred.append(pred.detach().cpu())
        return _estimation_loss

    def on_validation_epoch_end(self):
        _val_loss = self.val_loss_epoch / self.step
        self.log(
            "val_loss_epoch",
            _val_loss,
            prog_bar=True,
            on_epoch=True,
        )
        if _val_loss < self.best_val_loss:
            self.best_val_loss = _val_loss
            self.log(
                "best_val_loss",
                self.best_val_loss,
                prog_bar=True,
                on_epoch=True,
            )

        val_metrics = self._log_eval_metrics(
            prefix="val",
            label=self.val_label,
            pred=self.val_pred,
        )
        val_mae = float(val_metrics["val_mae_epoch"])
        val_outlier_fraction = float(val_metrics["val_outlier_fraction_epoch"])
        outlier_excess = max(0.0, val_outlier_fraction - self.sweep_outlier_max)
        val_sweep_score = val_mae + 0.2 * val_outlier_fraction + 1.0 * outlier_excess
        self.log(
            "val_sweep_score_epoch",
            val_sweep_score,
            prog_bar=True,
            on_epoch=True,
        )
        if val_sweep_score < self.best_val_sweep_score_epoch:
            self.best_val_sweep_score_epoch = val_sweep_score
            self.log(
                "best_val_sweep_score_epoch",
                self.best_val_sweep_score_epoch,
                prog_bar=True,
                on_epoch=True,
            )
        if val_mae < self.best_val_mae_epoch:
            self.best_val_mae_epoch = val_mae
            self.log(
                "best_val_mae_epoch",
                self.best_val_mae_epoch,
                prog_bar=True,
                on_epoch=True,
            )
        if val_outlier_fraction < self.best_val_outlier_fraction_epoch:
            self.best_val_outlier_fraction_epoch = val_outlier_fraction
            self.log(
                "best_val_outlier_fraction_epoch",
                self.best_val_outlier_fraction_epoch,
                prog_bar=True,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            flux,
            south_north_flag,
        ) = batch
        pred, _, _ = self.model(
            photometric=photometric,
            magnitudes=mags,
            output_spectrum=False,
        )
        _estimation_loss = self._get_estimation_loss(
            label=label,
            pred=pred,
            mode="test",
        )
        self.test_label.append(label.detach().cpu())
        self.test_pred.append(pred.detach().cpu())
        return _estimation_loss

    def on_test_epoch_end(self):
        self._log_eval_metrics(
            prefix="test",
            label=self.test_label,
            pred=self.test_pred,
        )

    def configure_optimizers(self):
        optimizer = bnb.optim.PagedAdamW8bit(
            self.model.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cos_annealing_t_0,
            T_mult=self.cos_annealing_t_mult,
            eta_min=self.cos_annealing_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "name": "lr"},
        }
