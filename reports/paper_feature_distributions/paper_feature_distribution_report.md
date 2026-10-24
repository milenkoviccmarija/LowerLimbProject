# Paper-Style Feature Distribution Figures

Input rows: 303,957
Phase bins: 20

## What the paper does

- TPDS is treated as a 2D Gaussian distribution for each locomotion mode and phase bin.
- KAT is treated as a 1D Gaussian distribution for each locomotion mode and phase bin.
- CPO is treated as a 2D Gaussian distribution for each locomotion mode.
- GRFPV is not modeled with a Gaussian in the same way; the paper uses geometry and Euclidean distance from GRF peak points to a line/circle representation.
- The probabilities from TPDS, KAT, CPO, and GRFPV are then used as classifier inputs.

## Generated figures

- `tpds_gaussian_phase_diagrams.png`
- `kat_gaussian_phase_trajectories.png`
- `cpo_gaussian_distribution.png`
- `grfpv_peak_geometry.png`

## Mode counts

| mode | rows |
| --- | --- |
| LW | 230071 |
| SA | 7827 |
| SD | 10762 |
| RA | 26192 |
| RD | 29105 |

## Dataset adaptation note

The upstream feature table uses thigh/shank IMU accelerometer pitch for segment-angle kinematics. When the dataset does not provide wearable heel/toe insole GRF, GRFPV is mapped to first/second stance vGRF peaks from native force signals, and the non-LW circles show a 95% distance radius for readability.
