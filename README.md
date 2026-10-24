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

`AB001.mat` is present but has not been integrated into the Python pipeline because it contains a MATLAB `MCOS` opaque `Signal` object rather than standard numeric arrays or tables.
