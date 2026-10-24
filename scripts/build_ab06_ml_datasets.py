"""Build the introductory AB06 ML datasets for this project.

The script is intentionally simple and inspectable. It does not implement the
final paper features yet. Instead, it builds a clean teaching baseline:

1. read the exported AB06 modalities listed in the manifest,
2. keep trials that have conditions, IMU, and force-plate data,
3. cut each trial into overlapping time windows,
4. assign each window the majority label from conditions,
5. calculate basic window statistics, and
6. save four sensor-set variants for model experiments.

The later paper-specific implementation can replace the statistics block with
TPDS, KAT, CPO, GRFPV, a standing classifier, and an ANN classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MANIFEST = Path("data/processed/ab06_modalities/export_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/ml_datasets")
DEFAULT_ACTIVITIES = ["levelground", "ramp", "stair"]

META_COLUMNS = {"subject", "session", "activity_folder", "trial_stem", "time_s"}
REQUIRED_MODALITIES = ("conditions", "imu", "fp")
STATS = ("mean", "std", "min", "max", "rms")
PAPER_STEADY_MODES = {"LW", "SA", "SD", "RA", "RD"}
PAPER_MODE_NAMES = {
    "LW": "level walking",
    "SA": "stair ascent",
    "SD": "stair descent",
    "RA": "ramp ascent",
    "RD": "ramp descent",
    "ST": "standing",
    "transition": "transition between modes",
    "turn": "turning",
    "treadmill": "treadmill walking",
    "unknown": "unknown",
}
PAPER_STAGE = "intro_window_stats"


@dataclass(frozen=True)
class DatasetVariant:
    """One output table definition."""

    name: str
    description: str
    imu_prefixes: tuple[str, ...]
    include_grf: bool
    paper_sensor_set: str


DATASET_VARIANTS = (
    DatasetVariant(
        name="thigh_shank_imu",
        description="two IMU segments only: thigh and shank",
        imu_prefixes=("thigh_", "shank_"),
        include_grf=False,
        paper_sensor_set="paper_like_two_imu",
    ),
    DatasetVariant(
        name="thigh_shank_imu_grf",
        description="paper-like starter set: thigh/shank IMU plus force-plate GRF",
        imu_prefixes=("thigh_", "shank_"),
        include_grf=True,
        paper_sensor_set="paper_like_two_imu_plus_force_plate_grf",
    ),
    DatasetVariant(
        name="all_imu",
        description="all available AB06 IMU segments: foot, shank, thigh, trunk",
        imu_prefixes=("foot_", "shank_", "thigh_", "trunk_"),
        include_grf=False,
        paper_sensor_set="ab06_all_imu",
    ),
    DatasetVariant(
        name="imu_grf",
        description="all available AB06 IMU segments plus force-plate GRF",
        imu_prefixes=("foot_", "shank_", "thigh_", "trunk_"),
        include_grf=True,
        paper_sensor_set="ab06_all_imu_plus_force_plate_grf",
    ),
)


def estimate_hz(time_s: pd.Series) -> float:
    diffs = np.diff(time_s.to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 200.0
    return float(1.0 / np.median(diffs))


def majority_label(labels: pd.Series) -> tuple[str, float]:
    counts = labels.dropna().astype(str).value_counts()
    if counts.empty:
        return "", 0.0
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())


def simplify_label(label: str) -> str:
    """Map dataset labels into a cleaner project label."""
    label = str(label)
    if "-" in label:
        return "transition"
    if label in {"idle", "stand"}:
        return "stand"
    if label == "walk":
        return "levelground"
    if label.startswith("turn"):
        return "turn"
    return label


def paper_mode_label(label: str) -> str:
    """Map AB06 labels to the six paper labels where that is justified."""
    label = str(label)
    if "-" in label:
        return "transition"
    if label in {"idle", "stand"}:
        return "ST"
    if label == "walk":
        return "LW"
    if label == "rampascent":
        return "RA"
    if label == "rampdescent":
        return "RD"
    if label == "stairascent":
        return "SA"
    if label == "stairdescent":
        return "SD"
    if label.startswith("turn"):
        return "turn"
    if label == "treadmill":
        return "treadmill"
    return "unknown"


def paper_classifier_stage(paper_mode: str) -> str:
    """Identify which later paper classifier stage should consume the row."""
    if paper_mode == "ST":
        return "standing_classifier"
    if paper_mode in PAPER_STEADY_MODES:
        return "ann_steady_state_classifier"
    return "excluded_from_paper_baseline"


def pair_manifest(manifest: pd.DataFrame, activities: list[str]) -> pd.DataFrame:
    expected = {
        "subject",
        "session",
        "activity_folder",
        "trial_stem",
        "modality",
        "output_path",
    }
    missing = expected - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest = manifest[manifest["activity_folder"].isin(activities)]
    paired = manifest.pivot_table(
        index=["subject", "session", "activity_folder", "trial_stem"],
        columns="modality",
        values="output_path",
        aggfunc="first",
    ).reset_index()
    paired.columns.name = None

    for modality in REQUIRED_MODALITIES:
        if modality not in paired.columns:
            paired[modality] = pd.NA

    paired = paired.dropna(subset=list(REQUIRED_MODALITIES))
    if paired.empty:
        raise ValueError(
            "No usable trials found. Need conditions, imu, and fp exports for at "
            f"least one of these activities: {activities}"
        )
    return paired


def label_imu_rows(imu: pd.DataFrame, conditions: pd.DataFrame, coarse_label: str) -> pd.Series:
    if "time_s" not in conditions.columns or "label" not in conditions.columns:
        return pd.Series([coarse_label] * len(imu), index=imu.index, dtype="string")

    left = imu[["time_s"]].sort_values("time_s")
    right = conditions[["time_s", "label"]].dropna(subset=["time_s"]).sort_values("time_s")
    if right.empty:
        return pd.Series([coarse_label] * len(imu), index=imu.index, dtype="string")

    merged = pd.merge_asof(left, right, on="time_s", direction="nearest")
    labels = merged["label"].fillna(coarse_label).astype("string")
    labels.index = left.index
    return labels.reindex(imu.index).fillna(coarse_label)


def numeric_columns(df: pd.DataFrame, prefixes: tuple[str, ...] | None = None) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in META_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if prefixes is not None and not column.startswith(prefixes):
            continue
        columns.append(column)
    return columns


def grf_columns(fp: pd.DataFrame) -> list[str]:
    """Use force components from force-plate data, not positions or moments."""
    columns: list[str] = []
    for column in fp.columns:
        if column in META_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(fp[column]):
            continue
        if column.endswith(("_vx", "_vy", "_vz")):
            columns.append(column)
    return columns


def add_stats(out: dict[str, Any], values: np.ndarray, columns: list[str], prefix: str) -> None:
    if values.size == 0 or not columns:
        return

    stat_values = {
        "mean": np.nanmean(values, axis=0),
        "std": np.nanstd(values, axis=0),
        "min": np.nanmin(values, axis=0),
        "max": np.nanmax(values, axis=0),
        "rms": np.sqrt(np.nanmean(np.square(values), axis=0)),
    }
    for stat_name in STATS:
        for column, value in zip(columns, stat_values[stat_name]):
            out[f"{prefix}{column}__{stat_name}"] = float(value)


def make_base_window_row(
    row: pd.Series,
    start_s: float,
    end_s: float,
    hz: float,
    window_n: int,
    n_fp_samples: int,
    label: str,
    label_purity: float,
) -> dict[str, Any]:
    """Create the non-feature columns shared by every dataset variant."""
    mode_label = simplify_label(label)
    paper_mode = paper_mode_label(label)
    return {
        "subject": row["subject"],
        "session": row["session"],
        "activity_folder": row["activity_folder"],
        "trial_stem": row["trial_stem"],
        "window_start_s": start_s,
        "window_end_s": end_s,
        "sample_rate_hz": hz,
        "n_imu_samples": window_n,
        "n_fp_samples": n_fp_samples,
        "label": label,
        "mode_label": mode_label,
        "label_purity": label_purity,
        "is_transition": "-" in str(label),
        "paper_mode_label": paper_mode,
        "paper_mode_name": PAPER_MODE_NAMES[paper_mode],
        "paper_classifier_stage": paper_classifier_stage(paper_mode),
        "paper_feature_stage": PAPER_STAGE,
    }


def make_variant_row(
    base: dict[str, Any],
    variant: DatasetVariant,
    imu_window: pd.DataFrame,
    imu_columns: list[str],
    fp_window: pd.DataFrame,
    grf_columns_: list[str],
) -> dict[str, Any]:
    """Add feature columns for one dataset variant."""
    out = base.copy()
    out["dataset_variant"] = variant.name
    out["paper_sensor_set"] = variant.paper_sensor_set

    add_stats(out, imu_window[imu_columns].to_numpy(dtype=np.float32), imu_columns, "imu_")
    if variant.include_grf:
        add_stats(out, fp_window.to_numpy(dtype=np.float32), grf_columns_, "grf_")
    return out


def rows_for_trial(row: pd.Series, window_s: float, step_s: float) -> list[dict[str, Any]]:
    imu = pd.read_parquet(row["imu"])
    fp = pd.read_parquet(row["fp"])
    conditions = pd.read_parquet(row["conditions"])

    labels = label_imu_rows(imu, conditions, row["activity_folder"])
    hz = estimate_hz(imu["time_s"])
    window_n = max(2, round(window_s * hz))
    step_n = max(1, round(step_s * hz))

    imu_columns_by_variant = {
        variant.name: numeric_columns(imu, prefixes=variant.imu_prefixes)
        for variant in DATASET_VARIANTS
    }
    grf_cols = grf_columns(fp)

    imu_times = imu["time_s"].to_numpy(dtype=float)
    fp_times = fp["time_s"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for start in range(0, len(imu) - window_n + 1, step_n):
        end = start + window_n
        start_s = float(imu_times[start])
        end_s = float(imu_times[end - 1])
        label, label_purity = majority_label(labels.iloc[start:end])

        fp_mask = (fp_times >= start_s) & (fp_times <= end_s)
        fp_window = fp.loc[fp_mask, grf_cols]
        imu_window = imu.iloc[start:end]

        base = make_base_window_row(
            row=row,
            start_s=start_s,
            end_s=end_s,
            hz=hz,
            window_n=window_n,
            n_fp_samples=int(fp_mask.sum()),
            label=label,
            label_purity=label_purity,
        )

        for variant in DATASET_VARIANTS:
            rows.append(
                make_variant_row(
                    base=base,
                    variant=variant,
                    imu_window=imu_window,
                    imu_columns=imu_columns_by_variant[variant.name],
                    fp_window=fp_window,
                    grf_columns_=grf_cols,
                )
            )

    return rows


def split_and_write_datasets(combined: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Write one CSV and one Parquet file per configured dataset variant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    for variant in DATASET_VARIANTS:
        dataset = combined[combined["dataset_variant"] == variant.name].copy()
        if dataset.empty:
            continue

        dataset = dataset.drop(columns=["dataset_variant"]).dropna(axis=1, how="all")
        datasets[variant.name] = dataset
        dataset.to_parquet(output_dir / f"{variant.name}.parquet", index=False)
        dataset.to_csv(output_dir / f"{variant.name}.csv", index=False)

    return datasets


def write_summary(datasets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    summary_rows = []
    variants_by_name = {variant.name: variant for variant in DATASET_VARIANTS}

    for name, dataset in datasets.items():
        classifier_counts = dataset["paper_classifier_stage"].value_counts()
        summary_rows.append(
            {
                "dataset": name,
                "description": variants_by_name[name].description,
                "rows": len(dataset),
                "columns": len(dataset.columns),
                "activities": ",".join(sorted(dataset["activity_folder"].unique())),
                "labels": ",".join(sorted(dataset["mode_label"].unique())),
                "paper_modes": ",".join(sorted(dataset["paper_mode_label"].unique())),
                "standing_rows": int(classifier_counts.get("standing_classifier", 0)),
                "steady_mode_rows": int(classifier_counts.get("ann_steady_state_classifier", 0)),
                "excluded_rows": int(classifier_counts.get("excluded_from_paper_baseline", 0)),
                "paper_feature_stage": PAPER_STAGE,
                "paper_sensor_set": ",".join(sorted(dataset["paper_sensor_set"].unique())),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)


def build_datasets(
    manifest_path: Path,
    output_dir: Path,
    activities: list[str],
    window_s: float,
    step_s: float,
    min_label_purity: float,
) -> dict[str, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive.")
    if not 0 <= min_label_purity <= 1:
        raise ValueError("min_label_purity must be between 0 and 1.")

    manifest = pd.read_csv(manifest_path)
    paired = pair_manifest(manifest, activities)

    rows: list[dict[str, Any]] = []
    for _, row in paired.iterrows():
        rows.extend(rows_for_trial(row, window_s=window_s, step_s=step_s))

    combined = pd.DataFrame(rows)
    if combined.empty:
        raise ValueError("No windows were created. Check window size, step size, and input trials.")

    combined = combined[combined["label_purity"] >= min_label_purity].copy()
    if combined.empty:
        raise ValueError(
            "All windows were removed by the label purity filter. "
            f"Try a lower --min-label-purity than {min_label_purity}."
        )

    datasets = split_and_write_datasets(combined, output_dir)
    write_summary(datasets, output_dir)
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activities", nargs="+", default=DEFAULT_ACTIVITIES)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--step-s", type=float, default=0.5)
    parser.add_argument("--min-label-purity", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = build_datasets(
        args.manifest,
        args.output_dir,
        args.activities,
        args.window_s,
        args.step_s,
        args.min_label_purity,
    )
    print(f"Output directory: {args.output_dir}")
    for variant, dataset in datasets.items():
        print(f"{variant}: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
        print(dataset["mode_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
