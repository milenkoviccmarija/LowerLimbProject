# Intelligent Prosthetic Knee Project

This project uses the AB06 dataset and the *Novel Feature Extraction and Locomotion Mode Classification Using Intelligent Lower-Limb Prosthesis* paper (Machines, 2023) as the main reference.

## Processing Pipeline

1. **Dataset Indexing** — `build_dataset_index.py`  
   Indexes MATLAB trial files and identifies available data modalities.

2. **MATLAB Data Export** — `export_ab06_modalities.py`  
   Exports MATLAB tables into Parquet files for Python processing.

3. **Baseline ML Dataset** — `build_ab06_ml_datasets.py`  
   Extracts window-based statistical features from IMU and force-plate signals.

4. **Paper Input Dataset** — `build_ab06_paper_input_dataset.py`  
   Prepares sample-level data for six locomotion modes: `ST`, `LW`, `SA`, `SD`, `RA`, and `RD`.

5. **Feature Extraction** — `build_ab06_paper_features.py`  
   Generates `TPDS`, `KAT`, `CPO`, and `GRFPV` feature groups.

6. **Classifier Input Generation** — `build_paper_classifier_inputs.py`  
   Converts extracted features into probability-based classifier inputs.

7. **Feature Visualization** — `plot_ab06_paper_feature_distributions.py`  
   Generates feature-distribution plots and parameter summaries.

## Data

The pipeline currently uses the `AB06/` dataset.

## Feature Extraction

The feature extraction pipeline combines IMU signals, force-plate measurements, and gait-cycle information to generate four feature groups: `TPDS`, `KAT`, `CPO`, and `GRFPV`.

### 1. Input Signals

For each trial, the pipeline processes:

- `imu` — thigh and shank sensor measurements
- `fp` — ground reaction force measurements
- `conditions` — locomotion mode labels
- `gcLeft` / `gcRight` — gait-cycle information

The paper-specific pipeline uses IMU-derived angles rather than `gon` or `ik` signals.

### 2. Angle Estimation

Thigh and shank angles are estimated from accelerometer measurements:

- `theta_t` — thigh pitch from `thigh_Accel_X/Y/Z`
- `theta_s` — shank pitch from `shank_Accel_X/Y/Z`
- `dot_theta_t` — rate of change of the thigh angle
- `theta_k` — estimated knee angle, calculated as:

```text
theta_k = theta_s - theta_t
```

### 3. Gait-Cycle Segmentation

The `HeelStrike` signal from `gcLeft` or `gcRight` represents gait-cycle progression from 0% to 100%. A transition from approximately 100% back to 0% indicates the beginning of a new cycle.

This segmentation allows the signals to be processed and normalized across individual gait cycles.

### 4. Feature Groups

**TPDS — Thigh Phase Diagram**

Describes thigh motion using the normalized thigh angle and its angular velocity. The phase angle is calculated as:

```text
phi_new = atan2(dot_theta_t_stn, theta_t_tn) + pi
```

Features:
- `theta_t_tn`
- `dot_theta_t_stn`
- `phi_new`

**KAT — Knee Angle Trajectory**

Describes knee motion throughout the gait cycle. The estimated knee angle is smoothed and normalized to produce:

- `theta_k_sn`

**CPO — Center-Point Offset**

Describes the displacement of the normalized thigh phase diagram using:

- `beta_x`
- `beta_y`

**GRFPV — Ground Reaction Force Peak Values**

Uses vertical force-plate measurements during the stance phase. The stance phase is divided into two portions, and the corresponding force peaks are extracted:

- `Ghp` — peak from the first portion of stance
- `Gtp` — peak from the second portion of stance

### 5. Standing Mode

The `ST` (standing) mode is handled separately because it does not contain a regular gait cycle. Instead of the gait-cycle-based `TPDS`, `KAT`, and `CPO` features, the pipeline uses `dynamic_trend` and `GRFPV`.

### Feature Summary

| Feature group | Main purpose | Output features |
|---|---|---|
| `TPDS` | Thigh motion pattern | `theta_t_tn`, `dot_theta_t_stn`, `phi_new` |
| `KAT` | Knee angle trajectory | `theta_k_sn` |
| `CPO` | Phase-diagram displacement | `beta_x`, `beta_y` |
| `GRFPV` | Ground reaction force peaks | `Gtp`, `Ghp` |
