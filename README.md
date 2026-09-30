# Helio_TAF_26

A research-oriented PV power forecasting project based on sky-image sequences, weather features, and temporal fusion.

## Overview

This repository provides:
- Data loading pipelines for HDF5-based image and meteorological sequences.
- Three model variants:
  - `PVNET_ONLY`: image branch only
  - `PVNET`: image + numeric feature fusion
  - `TAF`: temporal attention fusion (TAF layer)
- Cross-validation training and ensemble-style evaluation scripts.

## Project Structure

```text
.
|-- models/
|   |-- pvnet_with_lstm_t_timesteps.py
|   |-- finetune_10_pvnet_with_all_ra_lstm_t_timesteps.py
|   |-- finetune_taf_10_pvnet_with_all_ra_lstm_t_timesteps.py
|   `-- taf.py
|-- data_loader.py
|-- train.py
|-- test.py
|-- utils.py
|-- requirements.txt
`-- .gitignore
```

## Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Data and Checkpoints

By default, scripts expect:
- Data directory: `./data`
- Output directory: `./outputs`
- Pretrained PVNet checkpoint: `./checkpoints/pvnet/best_model.h5`

You can override them with environment variables:
- `PVNET_DATA_DIR`
- `PVNET_OUTPUT_DIR`
- `PRETRAINED_PVNET_PATH`

Example (PowerShell):

```powershell
$env:PVNET_DATA_DIR = "D:\\your_data_dir"
$env:PVNET_OUTPUT_DIR = "D:\\your_output_dir"
$env:PRETRAINED_PVNET_PATH = "D:\\your_ckpt\\best_model.h5"
```

## Expected Data Files

Training script (`train.py`) expects in `PVNET_DATA_DIR`:
- `train_2019_for_15.h5`
- `sampled_times_full_train_time_2019.npy`

Test script (`test.py`) expects:
- `test_2018_sampled_for_15.h5`
- `test_time_2018.npy`

## HDF5 Keys

The current code uses these dataset keys (must match your HDF5 exactly):
- `extracted_images_log`
- `pv_values`
- Weather feature keys such as:
  - `temperature_2m (掳C)`
  - `relative_humidity_2m (%)`
  - `dew_point_2m (掳C)`
  - `wind_speed_10m (km_h)`
  - etc.

Note: the degree symbol in your source data keys appears as `掳` due encoding history; keep keys unchanged unless you regenerate data.

## Train

Set `MODEL_TYPE` in `train.py`:
- `"PVNET_ONLY"`
- `"PVNET"`
- `"TAF"`

Then run:

```bash
python train.py
```

Outputs are saved under:
- `outputs/model_output/Fusion-PVnet-for-15/`

## Test

```bash
python test.py
```

The script:
- Loads all fold checkpoints (`repetition_1` ... `repetition_5`)
- Evaluates fold-wise performance
- Produces ensemble prediction and RMSE
- Writes combination metrics to `output_results.txt`

## Reproducibility Notes

- Random seeds are partially fixed in data splitting utilities.
- This project is script-based (not packaged) to keep parity with experiment workflows.

## Citation

If you use this code in research, please cite your corresponding paper and describe any dataset/key adaptations.
