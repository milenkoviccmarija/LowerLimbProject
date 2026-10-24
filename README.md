# Intelligent Prosthetic Knee Project

## Overview

This project explores locomotion mode classification for an intelligent lower-limb prosthesis using the **AB06 dataset** and the following research paper as the main algorithmic reference:

*Novel Feature Extraction and Locomotion Mode Classification Using Intelligent Lower-Limb Prosthesis* (Machines, 2023).

The pipeline covers dataset indexing, MATLAB data export, feature extraction, classifier input generation, and feature-distribution visualization.

## Dataset and Data Availability

The current pipeline uses the usable `AB06/` dataset.

The file `AB001.mat` is also present in the project, but it is not currently suitable for direct Python analysis because it contains a MATLAB `MCOS` opaque `Signal` object rather than standard numeric arrays or tables.

**Important:** `AB001.mat` should not be used for analysis until its contents have been exported from MATLAB into a Python-compatible format.

## Project Workflow

The recommended order for exploring and running the project is described below.

### 1. Dataset Indexing

**Script:** `scripts/build_dataset_index.py`

This script:
- Scans the raw `AB06/` directory for `.mat` trial files.
- Reads trial conditions.
- Records the available modalities for each trial.

Run:

```bash
python3 scripts/build_dataset_index.py
```

**Output:**

```text
data/processed/ab06_trial_index.csv
```

**Key concepts to understand:**
- Trial naming conventions.
- The meaning of `activity_folder`, `trial_stem`, and modality paths.
- The available raw signals: `imu`, `fp`, `conditions`, `gcLeft`, and `gcRight`.

### 2. MATLAB Data Export

**Script:** `scripts/export_ab06_modalities.py`

This script reads the dataset index, extracts MATLAB tables, normalizes column names (for example, `Header` to `time_s`), and exports each trial and modality to Parquet format.

Run:

```bash
python3 scripts/export_ab06_modalities.py --overwrite
```

**Output:**

```text
data/processed/ab06_modalities/
data/processed/ab06_modalities/export_manifest.csv
```

**Key concepts to understand:**
- Converting MATLAB tables into Python-readable data.
- The role of the export manifest as the main input for downstream scripts.

### 3. Baseline Machine Learning Dataset

**Script:** `scripts/build_ab06_ml_datasets.py`

This script generates baseline datasets using window-based statistical features extracted from IMU and force-plate signals.

Run:

```bash
python3 scripts/build_ab06_ml_datasets.py
```

**Output:**

```text
data/processed/ml_datasets/
```

**Key concepts to understand:**
- Assigning labels to time windows.
- Extracting basic features from sensor signals.
- The difference between a baseline approach and the paper-specific pipeline.

### 4. Paper-Specific Input Dataset

**Script:** `scripts/build_ab06_paper_input_dataset.py`

This script creates sample-level inputs adapted to the locomotion modes and signal definitions used in the reference paper.

The supported modes are:

- `ST` — Standing
- `LW` — Level Walking
- `SA` — Stair Ascent
- `SD` — Stair Descent
- `RA` — Ramp Ascent
- `RD` — Ramp Descent

The extracted signals and variables include:

- `theta_t`
- `dot_theta_t`
- `theta_s`
- `Gh` and `Gt`
- Gait phase
- First- and second-stance GRF peaks

Run:

```bash
python3 scripts/build_ab06_paper_input_dataset.py
```

**Output:**

```text
data/processed/paper_input_datasets/paper_input_samples.csv
data/processed/paper_input_datasets/summary.csv
```

**Key concepts to understand:**
- Adapting AB06 signals to the reference paper's methodology.
- The interpretation of thigh and shank angles.
- The limitations of using force-plate ground reaction force (GRF) as a proxy for heel- and toe-specific insole measurements.

### 5. Paper-Inspired Feature Extraction

**Script:** `scripts/build_ab06_paper_features.py`

This script transforms the paper input samples into four feature groups:

| Feature group | Description |
|---|---|
| `TPDS` | Thigh phase diagram features |
| `KAT` | Knee angle trajectory features |
| `CPO` | Center-point offset features |
| `GRFPV` | Ground reaction force peak-value features |

Steady locomotion modes are processed separately from standing.

Run:

```bash
python3 scripts/build_ab06_paper_features.py
```

**Output:**

```text
data/processed/paper_features/paper_features_like_paper.csv
data/processed/paper_features/paper_steady_features_like_paper.csv
```

**Key concepts to understand:**
- The meaning of each feature group.
- The use of gait-phase bins.
- Why standing (`ST`) requires separate handling from the `TPDS`, `KAT`, and `CPO` features.

### 6. Classifier Input Generation

**Script:** `scripts/build_paper_classifier_inputs.py`

This script transforms the extracted feature tables into classifier-ready probability inputs using:

- 2D Gaussian models for `TPDS`.
- 1D Gaussian models for `KAT`.
- 2D Gaussian models for `CPO`.
- Line/circle geometry for `GRFPV`.

Run:

```bash
python3 scripts/build_paper_classifier_inputs.py
```

**Output:**

```text
data/processed/paper_classifier_inputs/paper_ann_classifier_inputs.csv
```

**Key concepts to understand:**
- Converting extracted features into probability-based inputs.
- How these inputs approximate the feature-processing approach described in the reference paper.

### 7. Feature Distribution Visualization

**Script:** `scripts/plot_ab06_paper_feature_distributions.py`

This script generates feature-distribution plots inspired by the reference paper and saves summaries of the fitted Gaussian and geometric parameters.

Run:

```bash
python3 scripts/plot_ab06_paper_feature_distributions.py
```

**Output:**

```text
reports/paper_feature_distributions/
```

**Key concepts to understand:**
- Feature distributions for each locomotion mode.
- The degree of separation or overlap between modes.
- The suitability of the extracted features for classification.

## Core Scripts

The following scripts form the main processing pipeline and should be retained:

```text
scripts/build_dataset_index.py
scripts/export_ab06_modalities.py
scripts/build_ab06_ml_datasets.py
scripts/build_ab06_paper_input_dataset.py
scripts/build_ab06_paper_features.py
scripts/build_paper_classifier_inputs.py
scripts/plot_ab06_paper_feature_distributions.py
```

## Notes and Limitations

- The pipeline is based on the available AB06 data and is inspired by the methodology described in the Machines (2023) paper.
- Force-plate GRF measurements are not equivalent to separate heel and toe insole measurements.
- The extracted features and classifier inputs should be validated against the reference paper before claiming an exact reproduction of its methodology.
- `AB001.mat` requires an additional MATLAB export step before it can be used reliably in the Python pipeline.

## Expected Data Structure

A Python-compatible exported dataset should contain numeric tables with variables such as:

```text
time_s
theta_t
dot_theta_t
theta_s
Gh
Gt
label / mode
```

Alternatively, the exported data may be organized into modality-specific tables corresponding to:

```text
conditions
imu
fp
gcLeft
gcRight
```

The exported data should preserve the original timestamps, signal units, and locomotion labels so that the processing pipeline can use them consistently.
