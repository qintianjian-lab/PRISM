import lightning
import torch
from timm import optim

from model.loss import (
    NLLLoss,
    delta_z,
    nmad_z,
    sigma_n,
    outline_fraction,
    z_estimate,
    SpectraReconstructionLoss,
    PhotoReconstructionLoss,
)
from model.modules import BuildModel


class BuildLightningModel(lightning.LightningModule):

    def __init__(
        self,
        random_seed: int,
        eps: float | int,
        # hyper params
        learn_rate: float,
        cos_annealing_t_0: int,
        cos_annealing_t_mult: int,
        cos_annealing_eta_min: float,
        dropout: float,
        weight_decay: float,
        # photometric & mags settings
        photo_in_channel: int,
        photo_in_dim: int,
        mag_in_channel: int,
        mag_in_dim: int,
        out_channel: int,
        # spectrum settings
        spectrum_extractor_config: dict,
        enable_spectrum_auxiliary: bool,
        # PyTorch2.0 settings
        enable_torch_2: bool = False,
    ):
        super().__init__()
        print("[INFO] Using Random Seed: ", random_seed)
        self.model = (
            BuildModel(
                photo_in_channel=photo_in_channel,
                photo_in_dim=photo_in_dim,
                mag_in_channel=mag_in_channel,
                mag_in_dim=mag_in_dim,
                out_channel=out_channel,
                dropout=dropout,
                enable_spectrum=enable_spectrum_auxiliary,
                spectrum_extractor_config=spectrum_extractor_config,
                eps=eps,
            )
            if not enable_torch_2
            else torch.compile(
                model=BuildModel(
                    photo_in_channel=photo_in_channel,
                    photo_in_dim=photo_in_dim,
                    mag_in_channel=mag_in_channel,
                    mag_in_dim=mag_in_dim,
                    out_channel=out_channel,
                    dropout=dropout,
                    enable_spectrum=enable_spectrum_auxiliary,
                    spectrum_extractor_config=spectrum_extractor_config,
                    eps=eps,
                ),
                backend="inductor",
            )
        )
        if enable_torch_2:
            print("[INFO] Using PyTorch 2.0 compile")
        self.learn_rate = learn_rate
        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.weight_decay = weight_decay
        self.estimation_loss = NLLLoss(temperature=2.5, eps=eps)
        self.sr_loss = PhotoReconstructionLoss(photo_size=photo_in_dim)
        self.contrastive_loss = SpectraReconstructionLoss(eps=eps)
        self.contrastive_loss_weight_foreach_layer = [0.2, 0.4, 0.8, 1.0]
        self.enable_spectrum_auxiliary = enable_spectrum_auxiliary

        self.val_loss = torch.zeros(1).to(self.device)
        self.val_label = []
        self.val_pred = []
        self.val_loss_epoch = 0.0
        self.best_val_loss = 9e10
        self.best_val_mae_epoch = 9e10
        self.best_val_outline_fraction_epoch = 1.0

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
            ("sigma_0.05", sigma_n(_d_z, n=0.05)),
            ("sigma_0.15", sigma_n(_d_z, n=0.15)),
            ("outline_fraction", outline_fraction(_d_z, threshold=0.1)),
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
        _contrastive_loss = None
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            label,
            wavelength,
            flux,
        ) = batch
        if self.enable_spectrum_auxiliary:
            (
                pred,
                spectrum_features,
                sr_features,
                deep_features,
                photometric_features,
            ) = self.model(
                photometric=photometric,
                magnitudes=mags,
                output_spectrum=True,
            )
            _contrastive_loss = torch.tensor(0.0, device=self.device)
            for _index, _spec_feature in enumerate(spectrum_features):
                _deep_contrastive_loss = self.contrastive_loss(
                    _spec_feature,
                    flux,
                )
                self.log(
                    f"train_deep_contrastive_loss_{_index + 1}",
                    _deep_contrastive_loss,
                    prog_bar=True,
                    on_step=True,
                )
                _contrastive_loss += (
                    _deep_contrastive_loss
                    * self.contrastive_loss_weight_foreach_layer[_index]
                )
            self.log(
                "train_contrastive_loss_avg",
                _contrastive_loss,
                prog_bar=True,
                on_step=True,
            )
        else:
            (
                pred,
                spectrum_features,
                sr_features,
                deep_features,
                photometric_features,
            ) = self.model(
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
        _sr_loss = self.sr_loss(sr_features, photometric)
        self.log(
            "train_sr_loss",
            _sr_loss,
            prog_bar=True,
            on_step=True,
        )
        _estimation_loss = 0.81 * _estimation_loss + 0.19 * (10 * _sr_loss)
        self._log_eval_metrics(
            "train",
            label=label.detach().cpu(),
            pred=pred.detach().cpu(),
        )
        if self.enable_spectrum_auxiliary:
            return 0.5 * _estimation_loss + 0.5 * _contrastive_loss
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
            wavelength,
            flux,
        ) = batch
        (
            pred,
            spectrum_features,
            sr_features,
            deep_features,
            photometric_features,
        ) = self.model(
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
        if val_metrics["val_mae_epoch"] < self.best_val_mae_epoch:
            self.best_val_mae_epoch = val_metrics["val_mae_epoch"]
            self.log(
                "best_val_mae_epoch",
                self.best_val_mae_epoch,
                prog_bar=True,
                on_epoch=True,
            )
        if (
            val_metrics["val_outline_fraction_epoch"]
            < self.best_val_outline_fraction_epoch
        ):
            self.best_val_outline_fraction_epoch = val_metrics[
                "val_outline_fraction_epoch"
            ]
            self.log(
                "best_val_outline_fraction_epoch",
                self.best_val_outline_fraction_epoch,
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
            wavelength,
            flux,
        ) = batch
        (
            pred,
            spectrum_features,
            sr_features,
            deep_features,
            photometric_features,
        ) = self.model(
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
        optimizer = optim.create_optimizer_v2(
            self.model,
            opt="lion",
            lr=self.learn_rate,
            eps=self.eps,
            weight_decay=self.weight_decay,
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
