# Intelligent Prosthetic Knee Project

This project currently uses the usable `AB06/` dataset and the Machines 2023
paper as the main algorithmic reference:

```text
Novel Feature Extraction and Locomotion Mode Classification
Using Intelligent Lower-Limb Prosthesis
```

The file `AB001.mat` is present, but it is not directly usable from Python as a
dataset because it contains a MATLAB `MCOS` opaque `Signal` object, not normal
tables or arrays. Do not build analysis on it until it is exported from MATLAB
to plain numeric tables.

## What To Learn First

Read and run the project in this order.

### 1. Dataset Index

Script:

```text
scripts/build_dataset_index.py
```

Purpose:

- walks through the raw `AB06/` folder
- finds every `.mat` trial file
- reads `conditions`
- records which modalities exist for each trial

Run:

```bash
python3 scripts/build_dataset_index.py
```

Output:

```text
data/processed/ab06_trial_index.csv
```

Learn here:

- how trials are named
- what `activity_folder`, `trial_stem`, and modality paths mean
- which raw signals exist; the paper pipeline uses `imu`, `fp`, `conditions`, `gcLeft`, and `gcRight`

### 2. MATLAB To Parquet Export

Script:

```text
scripts/export_ab06_modalities.py
```

Purpose:

- reads the trial index
- opens MATLAB table files
- normalizes columns like `Header -> time_s`
- writes one Parquet file per trial and modality

Run:

```bash
python3 scripts/export_ab06_modalities.py --overwrite
```

Output:

```text
data/processed/ab06_modalities/
data/processed/ab06_modalities/export_manifest.csv
```

Learn here:

- how raw `.mat` tables become Python-readable tables
- why the manifest is the central input for later scripts

### 3. Simple ML Baseline

Script:

```text
scripts/build_ab06_ml_datasets.py
```

Purpose:

- makes simple windowed statistics from IMU and force-plate data
- creates baseline datasets before the paper-specific feature extraction

Run:

```bash
python3 scripts/build_ab06_ml_datasets.py
```

Output:

```text
data/processed/ml_datasets/
```

Learn here:

- how labels are attached to time windows
- how sensor columns become basic features
- why this is a baseline, not the final Machines-paper method

### 4. Paper Input Dataset

Script:

```text
scripts/build_ab06_paper_input_dataset.py
```

Purpose:

- builds sample-level paper inputs
- maps dataset labels into paper modes: `ST`, `LW`, `SA`, `SD`, `RA`, `RD`
- extracts paper-style signals:
  - `theta_t`
  - `dot_theta_t`
  - `theta_s`
  - `Gh`, `Gt`
  - gait phase
  - first/second stance GRF peaks

Run:

```bash
python3 scripts/build_ab06_paper_input_dataset.py
```

Output:

```text
data/processed/paper_input_datasets/paper_input_samples.csv
data/processed/paper_input_datasets/summary.csv
```

Learn here:

- how AB06 signals are adapted to the Machines paper
- why `theta_t` and `theta_s` are taken from thigh/shank IMU segment pitch
- why force-plate GRF is only a proxy for the paper's heel/toe insole GRF

### 5. Paper Feature Tables

Script:

```text
scripts/build_ab06_paper_features.py
```

Purpose:

- converts paper input samples into named feature groups:
  - `TPDS`
  - `KAT`
  - `CPO`
  - `GRFPV`
- keeps steady modes separate from standing logic

Run:

```bash
python3 scripts/build_ab06_paper_features.py
```

Output:

```text
data/processed/paper_features/paper_features_like_paper.csv
data/processed/paper_features/paper_steady_features_like_paper.csv
```

Learn here:

- what each final paper feature column means
- why phase bins are used
- why `ST` should not be mixed into `TPDS/KAT/CPO`

### 6. Classifier Inputs From Paper Features

Script:

```text
scripts/build_paper_classifier_inputs.py
```

Purpose:

- turns feature tables into classifier-ready probability inputs
- fits:
  - 2D Gaussian models for `TPDS`
  - 1D Gaussian models for `KAT`
  - 2D Gaussian models for `CPO`
  - line/circle geometry for `GRFPV`

Run:

```bash
python3 scripts/build_paper_classifier_inputs.py
```

Output:

```text
data/processed/paper_classifier_inputs/paper_ann_classifier_inputs.csv
```

Learn here:

- how raw paper features become probabilities
- why this is closer to the paper classifier than raw feature columns alone

### 7. Paper-Style Figures

Script:

```text
scripts/plot_ab06_paper_feature_distributions.py
```

Purpose:

- plots distributions similar to the paper's feature-analysis figures
- writes Gaussian/geometric parameter summaries

Run:

```bash
python3 scripts/plot_ab06_paper_feature_distributions.py
```

Output:

```text
reports/paper_feature_distributions/
```

Learn here:

- what the feature distributions look like by mode
- whether modes separate cleanly or overlap

## Scripts That Should Remain

```text
scripts/build_dataset_index.py
scripts/export_ab06_modalities.py
scripts/build_ab06_ml_datasets.py
scripts/build_ab06_paper_input_dataset.py
scripts/build_ab06_paper_features.py
scripts/build_paper_classifier_inputs.py
scripts/plot_ab06_paper_feature_distributions.py
```

## Important Note About AB001.mat

`AB001.mat` should not be used as the dataset yet. It needs a MATLAB export step
first. A useful exported dataset should contain plain numeric tables with columns
such as:

```text
time_s
theta_t
dot_theta_t
theta_s
Gh
Gt
label / mode
```

or modality tables equivalent to:

```text
conditions
imu
fp
gcLeft
gcRight
```
