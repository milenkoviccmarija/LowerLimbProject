"""Build model-ready features inspired by the main Machines paper.

Input:
    data/processed/paper_input_datasets/paper_input_samples.csv

Outputs:
    data/processed/paper_features/paper_features_like_paper.csv
    data/processed/paper_features/paper_steady_features_like_paper.csv

The main paper-style representation is kept as four feature groups:

    TPDS  -> normalized thigh phase diagram and phase
    KAT   -> normalized knee angle trajectory proxy
    CPO   -> phase-diagram center-point offset
    GRFPV -> peak vertical GRF proxies
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/processed/paper_input_datasets/paper_input_samples.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/paper_features")
DEFAULT_PHASE_BINS = 20
PAPER_STEADY_MODES = {"LW", "SA", "SD", "RA", "RD"}
PAPER_INPUT_MODES = {"ST", *PAPER_STEADY_MODES}

META_COLUMNS = [
    "subject",
    "session",
    "activity_folder",
    "trial_stem",
    "time_s",
    "label",
    "omega_m",
    "paper_mode_name",
]

PAPER_GROUP_COLUMNS = [
    ("theta_t_tn", "tpds_theta_t_tn"),
    ("dot_theta_t_stn", "tpds_dot_theta_t_stn"),
    ("phi_new", "tpds_phi_new"),
    ("theta_k_sn", "kat_theta_k_sn"),
    ("beta_x", "cpo_beta_x"),
    ("beta_y", "cpo_beta_y"),
    ("Gtp", "grfpv_Gtp"),
    ("Ghp", "grfpv_Ghp"),
]
PAPER_GROUP_RENAME = dict(PAPER_GROUP_COLUMNS)
PAPER_LIKE_FEATURES = [renamed for _, renamed in PAPER_GROUP_COLUMNS]
ST_GROUP_RENAME = {
    "dynamic_trend": "st_dynamic_trend",
    "Gtp": "grfpv_Gtp",
    "Ghp": "grfpv_Ghp",
}
ST_LIKE_FEATURES = list(ST_GROUP_RENAME.values())

def load_input(path: Path) -> pd.DataFrame:
    """Load only the columns needed to build the paper feature tables."""
    needed = [*META_COLUMNS, *PAPER_GROUP_RENAME.keys(), "dynamic_trend"]
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path, usecols=lambda column: column in needed)
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {sorted(missing)}")
    return df


def add_phase_bin(out: pd.DataFrame, phase_bins: int) -> pd.DataFrame:
    """Add the phase interval index used for paper-style feature distributions."""
    if phase_bins <= 0:
        raise ValueError("phase_bins must be positive.")

    phase = pd.to_numeric(out["tpds_phi_new"], errors="coerce")
    bin_width = 2.0 * np.pi / phase_bins
    bins = np.floor((phase % (2.0 * np.pi)) / bin_width)
    out["tpds_phi_bin"] = bins.clip(lower=0, upper=phase_bins - 1).astype("Int64")
    return out


def build_paper_like_features(df: pd.DataFrame, phase_bins: int) -> pd.DataFrame:
    """Keep the four paper features in their native grouped form."""
    ordered_input_columns = [original for original, _ in PAPER_GROUP_COLUMNS]
    out = df[[*META_COLUMNS, *ordered_input_columns, "dynamic_trend"]].copy()
    out = out.rename(columns={**PAPER_GROUP_RENAME, "dynamic_trend": "st_dynamic_trend"})
    out = add_phase_bin(out, phase_bins=phase_bins)
    out["feature_set"] = "paper_features_like_paper"
    out["feature_note"] = (
        "Paper-aligned features: steady modes use TPDS, KAT, CPO, and GRFPV; ST keeps only dynamic trend and GRFPV because ST has no stable phase variable."
    )
    out["paper_classifier_stage"] = np.select(
        [
            out["omega_m"].isin(PAPER_STEADY_MODES),
            out["omega_m"].eq("ST"),
        ],
        [
            "ann_steady_state_classifier",
            "standing_classifier_grfpv_only",
        ],
        default="excluded_from_paper_baseline",
    )
    return out


def build_features(
    input_path: Path,
    output_dir: Path,
    phase_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save all paper feature outputs."""
    df = load_input(input_path)
    paper_like = build_paper_like_features(df, phase_bins=phase_bins)
    paper_like = paper_like[paper_like["omega_m"].isin(PAPER_INPUT_MODES)].copy()
    steady_paper_like = paper_like[paper_like["omega_m"].isin(PAPER_STEADY_MODES)].dropna(
        subset=PAPER_LIKE_FEATURES
    ).copy()
    steady_paper_like["feature_set"] = "paper_steady_features_like_paper"
    standing_paper_like = paper_like[paper_like["omega_m"].eq("ST")].dropna(
        subset=ST_LIKE_FEATURES
    ).copy()
    standing_paper_like["feature_set"] = "paper_standing_features_like_paper"

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_like.to_csv(output_dir / "paper_features_like_paper.csv", index=False)
    paper_like.to_parquet(output_dir / "paper_features_like_paper.parquet", index=False)
    steady_paper_like.to_csv(output_dir / "paper_steady_features_like_paper.csv", index=False)
    steady_paper_like.to_parquet(
        output_dir / "paper_steady_features_like_paper.parquet",
        index=False,
    )
    standing_paper_like.to_csv(output_dir / "paper_standing_features_like_paper.csv", index=False)
    standing_paper_like.to_parquet(
        output_dir / "paper_standing_features_like_paper.parquet",
        index=False,
    )
    return paper_like, steady_paper_like, standing_paper_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase-bins", type=int, default=DEFAULT_PHASE_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_like, steady_paper_like, standing_paper_like = build_features(
        args.input,
        args.output_dir,
        phase_bins=args.phase_bins,
    )
    print(f"Output directory: {args.output_dir}")
    print(f"paper_features_like_paper: {paper_like.shape[0]} rows x {paper_like.shape[1]} columns")
    print(
        "paper_steady_features_like_paper: "
        f"{steady_paper_like.shape[0]} rows x {steady_paper_like.shape[1]} columns"
    )
    print(
        "paper_standing_features_like_paper: "
        f"{standing_paper_like.shape[0]} rows x {standing_paper_like.shape[1]} columns"
    )
    print(paper_like["omega_m"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
