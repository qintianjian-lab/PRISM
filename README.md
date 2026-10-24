# Photometric Redshift Estimation via Image and Photometry with Spectra-guided Gaussian Mixture Density Network

<div align="center">

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1538--3873%2Faeb072)](https://doi.org/10.1088/1538-3873/aeb072)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.x-792ee5)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

Accurate and computationally efficient photometric redshift estimation is essential for cosmological analyses and studies of the large-scale structure of the Universe. However, applying existing template-fitting and machine learning methods to rapidly growing survey data sets can involve trade-offs between predictive performance and computational throughput. Moreover, feature extractors developed for natural images may not be well suited to the sparse, source-centered, and multiband characteristics of galaxy images. We present PRISM (Photometric Redshift Estimation via a Spectra-guided Gaussian Mixture Density Framework), a multimodal framework that jointly exploits multiband image cutouts and catalog-level photometric measurements. PRISM incorporates feature-extraction modules tailored to source-centered astronomical images and a dedicated fusion module that captures complementary information from the imaging and tabular modalities. During training, a frozen spectral encoder provides an auxiliary representation-alignment objective, encouraging selected photometric features to retain information associated with galaxy spectra without requiring spectroscopic data at inference time. A Gaussian-mixture density head models the conditional redshift distribution, enabling PRISM to produce a full predictive probability density function in addition to a point estimate. We evaluated PRISM through five-fold cross-validation on 1.98 million galaxies constructed from the Dark Energy Spectroscopic Instrument Data Release 1 and the DESI Legacy Imaging Surveys. Within this benchmark, PRISM reduces the mean absolute error and catastrophic-outlier fraction by 17.1% and 20.0%, respectively, relative to the best baseline value for each metric. Our PRISM ablations showed that combining multiband images with catalog flux measurements improved predictive performance relative to either input alone. Spectra-guided supervision provided a smaller additional gain in the full multimodal configuration.

![PRISM architecture](img/model.png)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/qintianjian-lab/PRISM
cd PRISM
pip install -r requirements.txt
git submodule update --init --recursive
```

## Data Preparation

Set the dataset location in `config/config.py`:

```python
config["dataset_root_dir"] = "/path/to/your/dataset"
config["hdf5_file_name"] = "dataset.h5"
config["label_dir_name"] = "label"
```

For five-fold experiments, the expected label layout is:

```text
/path/to/your/dataset/
|-- dataset.h5
`-- label/
    |-- fold_0/
    |   |-- train.csv
    |   |-- val.csv
    |   `-- test.csv
    |-- fold_1/
    `-- ...
```

Each label CSV should contain the following required columns:

| Column             | Description                                                     |
|--------------------|-----------------------------------------------------------------|
| `TARGETID`         | DESI target identifier                                          |
| `TARGET_RA`        | Right ascension                                                 |
| `TARGET_DEC`       | Declination                                                     |
| `Z`                | Spectroscopic redshift label                                    |
| `h5_index`         | Row index in the HDF5 arrays                                    |
| `south_north_flag` | Source/domain flag used for grouped evaluation                  |
| `magnitudes*`      | Photometric magnitude features                                  |
| `magnitudes_diff*` | Color-difference features                                       |
| `h5_file`          | Optional per-row HDF5 path; if absent, `hdf5_file_name` is used |

The HDF5 file is expected to provide:

| Key           | Description                          |
|---------------|--------------------------------------|
| `photo_flat`  | Flattened image cutouts              |
| `photo_shape` | Original shape for each image cutout |
| `spec_flat`   | Flattened spectral arrays            |
| `spec_shape`  | Original shape for each spectrum     |

The dataloader reconstructs image cutouts from `photo_flat/photo_shape`, normalizes the image channels with
`photo_min_max`, resizes them to `photo_in_size`, concatenates magnitude and color features, and loads spectra for the
auxiliary alignment loss during training.

## Configuration

Main settings live in `config/config.py`.

Important options include:

| Key                                  | Purpose                                                   |
|--------------------------------------|-----------------------------------------------------------|
| `used_device`                        | GPU IDs available for training                            |
| `batch_size`, `epochs`, `precision`  | Training scale and numerical precision                    |
| `photo_in_channel`, `photo_in_size`  | Input image channel count and cutout size                 |
| `mag_in_size`                        | Magnitude plus color feature length                       |
| `out_channel`                        | Number of Gaussian mixture components                     |
| `enable_wandb`, `wandb_project_name` | Weights & Biases logging                                  |
| `spectrum_pretrained_ckpt_dir`       | Directory containing fold-specific SSCNN checkpoints      |
| `spectral_supervised_layers`         | PRISM feature stages aligned to spectral embeddings       |
| `loss_weights`                       | Mixture NLL temperature and spectra-alignment loss weight |

The default setup uses five Gaussian mixture components and expects DESI `g/r/z` image channels.

## Training

Train one fold:

```bash
python train.py --cross_validation_fold_name fold_0 --random_seed 42
```

Run a quick debug pass:

```bash
python train.py --cross_validation_fold_name fold_0 --debug
```

## Inference and Evaluation

`predict.py` contains the default five-fold checkpoint mapping in `PRED_CONFIG`:

```python
PRED_CONFIG = {
    "list": [
        {
            "model_name": "PRISM",
            "ckpt_name": {
                "fold_0": "...",
                "fold_1": "...",
                "fold_2": "...",
                "fold_3": "...",
                "fold_4": "...",
            },
        },
    ],
    "save_path": "./results/DATASET_1.9M",
}
```

Run inference with the configured checkpoints:

```bash
python predict.py --clear --infer-precision bf16
```

Use full precision inference if needed:

```bash
python predict.py --clear --infer-precision fp32
```

The script writes fold-level outputs and an all-fold summary under `results/`:

```text
results/DATASET_1.9M/PRISM/
|-- fold_0/
|   |-- result.csv
|   `-- result.json
|-- fold_1/
|-- ...
`-- all_folds/
    |-- result.csv
    `-- result.json
```

`result.csv` contains per-object predictions, residuals, likelihood metrics, CRPS values, and exported Gaussian mixture
parameters. `result.json` contains aggregate metrics, redshift-binned metrics, PIT calibration metrics, inference
throughput, and grouped metrics by `south_north_flag`.

## Model Summary

To inspect the configured network:

```bash
python summary.py
```

This runs `torchinfo.summary` with the current configuration.

## Citation

```bibtex
@article{He_2026,
doi = {10.1088/1538-3873/aeb072},
url = {https://doi.org/10.1088/1538-3873/aeb072},
year = {2026},
month = {oct},
publisher = {The Astronomical Society of the Pacific},
volume = {138},
number = {10},
pages = {104506},
author = {He, Yuchen and Wu, Jingjing and Jiang, Bin and Zhang, Yanxia},
title = {Photometric Redshift Estimation via a Spectra-guided Gaussian Mixture Density Framework},
journal = {Publications of the Astronomical Society of the Pacific},
}
```
