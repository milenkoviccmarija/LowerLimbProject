"""Create paper-style visualizations for local feature distributions.

The main paper presents the four features visually before using them for
classification:

    TPDS  -> phase-binned 2D Gaussian distributions
    KAT   -> phase-binned 1D Gaussian distributions
    CPO   -> 2D Gaussian distributions by mode
    GRFPV -> GRF peak geometry by mode

This script creates local-dataset versions of those figures from the
model-ready feature table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse

DEFAULT_DATASET = Path("data/processed/paper_features/paper_steady_features_like_paper.csv")
DEFAULT_OUTPUT_DIR = Path("reports/paper_feature_distributions")
PAPER_MODES = ["LW", "SA", "SD", "RA", "RD"]
MODE_COLORS = {
    "LW": "#2563eb",
    "SA": "#9333ea",
    "SD": "#ea580c",
    "RA": "#16a34a",
    "RD": "#dc2626",
}
FEATURE_COLUMNS = [
    "tpds_theta_t_tn",
    "tpds_dot_theta_t_stn",
    "tpds_phi_new",
    "tpds_phi_bin",
    "kat_theta_k_sn",
    "cpo_beta_x",
    "cpo_beta_y",
    "grfpv_Gtp",
    "grfpv_Ghp",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the steady-mode feature table and coerce feature columns to numeric."""
    columns = [
        "subject",
        "trial_stem",
        "time_s",
        "label",
        "omega_m",
        "paper_mode_name",
        *FEATURE_COLUMNS,
    ]
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    df = pd.read_csv(path, usecols=lambda column: column in columns)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing)}")

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    df["tpds_phi_bin"] = df["tpds_phi_bin"].astype(int)
    return df[df["omega_m"].isin(PAPER_MODES)].copy()


def sample_by_mode(df: pd.DataFrame, max_per_mode: int, random_state: int) -> pd.DataFrame:
    """Sample each mode for readable scatter plots."""
    parts = []
    for mode in PAPER_MODES:
        subset = df[df["omega_m"] == mode]
        if len(subset) > max_per_mode:
            subset = subset.sample(max_per_mode, random_state=random_state)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def covariance_ellipse(
    mean_x: float,
    mean_y: float,
    cov_xx: float,
    cov_xy: float,
    cov_yy: float,
    n_std: float,
) -> tuple[float, float, float] | None:
    """Return ellipse width, height, and angle for a 2D covariance matrix."""
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    if not np.isfinite(covariance).all():
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return None

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(eigenvalues)
    if not np.isfinite([mean_x, mean_y, width, height, angle]).all():
        return None
    return float(width), float(height), float(angle)


def add_covariance_ellipse(
    axis: plt.Axes,
    row: pd.Series,
    x_column: str,
    y_column: str,
    cov_xx_column: str,
    cov_xy_column: str,
    cov_yy_column: str,
    color: str,
    n_std: float,
    linewidth: float = 1.1,
    alpha: float = 0.30,
) -> None:
    """Draw one covariance ellipse if the covariance values are usable."""
    params = covariance_ellipse(
        row[x_column],
        row[y_column],
        row[cov_xx_column],
        row[cov_xy_column],
        row[cov_yy_column],
        n_std=n_std,
    )
    if params is None:
        return
    width, height, angle = params
    ellipse = Ellipse(
        (row[x_column], row[y_column]),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    axis.add_patch(ellipse)


def phase_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Calculate paper-style Gaussian parameters by mode and phase bin."""
    phase_bins = int(df["tpds_phi_bin"].max()) + 1
    rows = []
    for (mode, phase_bin), subset in df.groupby(["omega_m", "tpds_phi_bin"], sort=True):
        if len(subset) < 3:
            continue

        tpds_values = subset[["tpds_theta_t_tn", "tpds_dot_theta_t_stn"]].to_numpy(dtype=float)
        tpds_cov = np.cov(tpds_values, rowvar=False)
        kat_values = subset["kat_theta_k_sn"].to_numpy(dtype=float)
        rows.append(
            {
                "omega_m": mode,
                "tpds_phi_bin": int(phase_bin),
                "phase_center_rad": (float(phase_bin) + 0.5) * 2.0 * np.pi / phase_bins,
                "count": len(subset),
                "tpds_theta_mean": float(tpds_values[:, 0].mean()),
                "tpds_dot_theta_mean": float(tpds_values[:, 1].mean()),
                "tpds_cov_xx": float(tpds_cov[0, 0]),
                "tpds_cov_xy": float(tpds_cov[0, 1]),
                "tpds_cov_yy": float(tpds_cov[1, 1]),
                "kat_mean": float(kat_values.mean()),
                "kat_std": float(kat_values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows), phase_bins


def set_quantile_limits(axis: plt.Axes, x: pd.Series, y: pd.Series, q_low: float = 0.01) -> None:
    """Set readable scatter limits using quantiles."""
    q_high = 1.0 - q_low
    x_min, x_max = x.quantile([q_low, q_high])
    y_min, y_max = y.quantile([q_low, q_high])
    x_pad = max(1.0, 0.08 * float(x_max - x_min))
    y_pad = max(1.0, 0.08 * float(y_max - y_min))
    axis.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
    axis.set_ylim(float(y_min - y_pad), float(y_max + y_pad))


def save_tpds_figure(
    df: pd.DataFrame,
    phase_stats: pd.DataFrame,
    output_dir: Path,
    samples_per_mode: int,
) -> None:
    """Plot TPDS as phase-diagram clouds plus phase-binned Gaussian ellipses."""
    sample = sample_by_mode(df, max_per_mode=samples_per_mode, random_state=10)
    fig, axes = plt.subplots(1, len(PAPER_MODES), figsize=(17, 3.6), sharex=True, sharey=True)
    for axis, mode in zip(axes, PAPER_MODES):
        color = MODE_COLORS[mode]
        subset = sample[sample["omega_m"] == mode]
        stats = phase_stats[phase_stats["omega_m"] == mode].sort_values("tpds_phi_bin")

        axis.scatter(
            subset["tpds_theta_t_tn"],
            subset["tpds_dot_theta_t_stn"],
            s=3,
            alpha=0.16,
            color=color,
            linewidths=0,
        )
        axis.plot(
            stats["tpds_theta_mean"],
            stats["tpds_dot_theta_mean"],
            color="#111827",
            linewidth=1.6,
        )

        ellipse_rows = stats[stats["tpds_phi_bin"] % 3 == 0]
        for _, row in ellipse_rows.iterrows():
            add_covariance_ellipse(
                axis,
                row,
                "tpds_theta_mean",
                "tpds_dot_theta_mean",
                "tpds_cov_xx",
                "tpds_cov_xy",
                "tpds_cov_yy",
                color=color,
                n_std=1.0,
            )

        axis.axhline(0, color="#d1d5db", linewidth=0.8)
        axis.axvline(0, color="#d1d5db", linewidth=0.8)
        axis.set_title(mode)
        axis.set_xlabel("theta_t_tn")
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("dot_theta_t_stn")
    fig.suptitle("TPDS: thigh phase diagram shape with phase-binned Gaussian ellipses", y=1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "tpds_gaussian_phase_diagrams.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_kat_figure(phase_stats: pd.DataFrame, output_dir: Path) -> None:
    """Plot KAT as phase-binned normal mean plus one standard deviation."""
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    for mode in PAPER_MODES:
        subset = phase_stats[phase_stats["omega_m"] == mode].sort_values("tpds_phi_bin")
        if subset.empty:
            continue
        x = subset["phase_center_rad"].to_numpy(dtype=float)
        mean = subset["kat_mean"].to_numpy(dtype=float)
        std = subset["kat_std"].fillna(0.0).to_numpy(dtype=float)
        color = MODE_COLORS[mode]
        axis.plot(x, mean, color=color, linewidth=2.0, label=mode)
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.13)

    axis.set_xlim(0.0, 2.0 * np.pi)
    axis.set_xlabel("phi_new phase (rad)")
    axis.set_ylabel("kat_theta_k_sn")
    axis.set_title("KAT: normalized knee angle trajectory by gait phase")
    axis.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(output_dir / "kat_gaussian_phase_trajectories.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_cpo_figure(df: pd.DataFrame, output_dir: Path, samples_per_mode: int) -> pd.DataFrame:
    """Plot CPO scatter plus 95% Gaussian covariance ellipses."""
    sample = sample_by_mode(df, max_per_mode=samples_per_mode, random_state=21)
    rows = []
    for mode, subset in df.groupby("omega_m", sort=True):
        values = subset[["cpo_beta_x", "cpo_beta_y"]].to_numpy(dtype=float)
        covariance = np.cov(values, rowvar=False)
        rows.append(
            {
                "omega_m": mode,
                "count": len(subset),
                "cpo_beta_x_mean": float(values[:, 0].mean()),
                "cpo_beta_y_mean": float(values[:, 1].mean()),
                "cpo_cov_xx": float(covariance[0, 0]),
                "cpo_cov_xy": float(covariance[0, 1]),
                "cpo_cov_yy": float(covariance[1, 1]),
            }
        )
    cpo_stats = pd.DataFrame(rows)

    fig, axis = plt.subplots(figsize=(7.0, 5.4))
    for mode in PAPER_MODES:
        color = MODE_COLORS[mode]
        subset = sample[sample["omega_m"] == mode]
        axis.scatter(
            subset["cpo_beta_x"],
            subset["cpo_beta_y"],
            s=4,
            alpha=0.20,
            color=color,
            linewidths=0,
            label=mode,
        )
        stats = cpo_stats[cpo_stats["omega_m"] == mode]
        if not stats.empty:
            add_covariance_ellipse(
                axis,
                stats.iloc[0],
                "cpo_beta_x_mean",
                "cpo_beta_y_mean",
                "cpo_cov_xx",
                "cpo_cov_xy",
                "cpo_cov_yy",
                color=color,
                n_std=np.sqrt(5.991),
                linewidth=1.8,
                alpha=0.85,
            )

    set_quantile_limits(axis, sample["cpo_beta_x"], sample["cpo_beta_y"])
    axis.set_xlabel("cpo_beta_x")
    axis.set_ylabel("cpo_beta_y")
    axis.set_title("CPO: center position offset with Gaussian density contours")
    axis.legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(output_dir / "cpo_gaussian_distribution.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return cpo_stats


def save_grfpv_figure(df: pd.DataFrame, output_dir: Path, samples_per_mode: int) -> pd.DataFrame:
    """Plot GRFPV points with an LW line and non-LW distance circles."""
    nonzero = df[(df["grfpv_Ghp"] > 1e-9) & (df["grfpv_Gtp"] > 1e-9)].copy()
    sample = sample_by_mode(nonzero, max_per_mode=samples_per_mode, random_state=30)
    rows = []

    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for mode in PAPER_MODES:
        color = MODE_COLORS[mode]
        subset = sample[sample["omega_m"] == mode]
        axis.scatter(
            subset["grfpv_Ghp"],
            subset["grfpv_Gtp"],
            s=7,
            alpha=0.28,
            color=color,
            linewidths=0,
            label=mode,
        )

    lw = nonzero[nonzero["omega_m"] == "LW"]
    if len(lw) >= 2:
        slope, intercept = np.polyfit(lw["grfpv_Ghp"], lw["grfpv_Gtp"], deg=1)
        x_min, x_max = lw["grfpv_Ghp"].quantile([0.01, 0.99])
        x_line = np.linspace(float(x_min), float(x_max), 100)
        axis.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=2.1, label="LW fit")
        rows.append(
            {
                "omega_m": "LW",
                "geometry": "least_squares_line",
                "count_nonzero": len(lw),
                "line_slope": float(slope),
                "line_intercept": float(intercept),
                "circle_center_x": np.nan,
                "circle_center_y": np.nan,
                "circle_radius_95": np.nan,
            }
        )

    for mode in [mode for mode in PAPER_MODES if mode != "LW"]:
        subset = nonzero[nonzero["omega_m"] == mode]
        if subset.empty:
            continue
        center_x = float(subset["grfpv_Ghp"].median())
        center_y = float(subset["grfpv_Gtp"].median())
        distances = np.sqrt(
            np.square(subset["grfpv_Ghp"] - center_x) + np.square(subset["grfpv_Gtp"] - center_y)
        )
        radius = float(distances.quantile(0.95))
        circle = Circle(
            (center_x, center_y),
            radius,
            fill=False,
            edgecolor=MODE_COLORS[mode],
            linewidth=1.8,
            alpha=0.85,
        )
        axis.add_patch(circle)
        rows.append(
            {
                "omega_m": mode,
                "geometry": "median_center_95_percent_radius",
                "count_nonzero": len(subset),
                "line_slope": np.nan,
                "line_intercept": np.nan,
                "circle_center_x": center_x,
                "circle_center_y": center_y,
                "circle_radius_95": radius,
            }
        )

    set_quantile_limits(axis, sample["grfpv_Ghp"], sample["grfpv_Gtp"], q_low=0.01)
    axis.set_xlabel("grfpv_Ghp")
    axis.set_ylabel("grfpv_Gtp")
    axis.set_title("GRFPV: peak force proxy geometry by mode")
    axis.legend(frameon=False, markerscale=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "grfpv_peak_geometry.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    phase_stats: pd.DataFrame,
    cpo_stats: pd.DataFrame,
    grfpv_stats: pd.DataFrame,
    output_dir: Path,
    phase_bins: int,
) -> None:
    """Write a concise explanation of how the figures match the paper."""
    counts = df["omega_m"].value_counts().reindex(PAPER_MODES).fillna(0).astype(int)
    lines = [
        "# Paper-Style Feature Distribution Figures",
        "",
        f"Input rows: {len(df):,}",
        f"Phase bins: {phase_bins}",
        "",
        "## What the paper does",
        "",
        "- TPDS is treated as a 2D Gaussian distribution for each locomotion mode and phase bin.",
        "- KAT is treated as a 1D Gaussian distribution for each locomotion mode and phase bin.",
        "- CPO is treated as a 2D Gaussian distribution for each locomotion mode.",
        "- GRFPV is not modeled with a Gaussian in the same way; the paper uses geometry and Euclidean distance from GRF peak points to a line/circle representation.",
        "- The probabilities from TPDS, KAT, CPO, and GRFPV are then used as classifier inputs.",
        "",
        "## Generated figures",
        "",
        "- `tpds_gaussian_phase_diagrams.png`",
        "- `kat_gaussian_phase_trajectories.png`",
        "- `cpo_gaussian_distribution.png`",
        "- `grfpv_peak_geometry.png`",
        "",
        "## Mode counts",
        "",
        "| mode | rows |",
        "| --- | --- |",
    ]
    for mode, count in counts.items():
        lines.append(f"| {mode} | {count} |")
    lines.extend(
        [
            "",
            "## Dataset adaptation note",
            "",
            "The upstream feature table uses thigh/shank IMU accelerometer pitch for segment-angle kinematics. When the dataset does not provide wearable heel/toe insole GRF, GRFPV is mapped to first/second stance vGRF peaks from native force signals, and the non-LW circles show a 95% distance radius for readability.",
        ]
    )

    (output_dir / "paper_feature_distribution_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    phase_stats.to_csv(output_dir / "tpds_kat_phase_gaussian_parameters.csv", index=False)
    cpo_stats.to_csv(output_dir / "cpo_gaussian_parameters.csv", index=False)
    grfpv_stats.to_csv(output_dir / "grfpv_geometry_parameters.csv", index=False)


def build_figures(
    dataset_path: Path,
    output_dir: Path,
    samples_per_mode: int,
) -> None:
    """Build all paper-style distribution figures and parameter tables."""
    df = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_stats, phase_bins = phase_statistics(df)
    save_tpds_figure(df, phase_stats, output_dir, samples_per_mode=samples_per_mode)
    save_kat_figure(phase_stats, output_dir)
    cpo_stats = save_cpo_figure(df, output_dir, samples_per_mode=samples_per_mode)
    grfpv_stats = save_grfpv_figure(df, output_dir, samples_per_mode=samples_per_mode)
    write_report(df, phase_stats, cpo_stats, grfpv_stats, output_dir, phase_bins=phase_bins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-mode", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figures(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        samples_per_mode=args.samples_per_mode,
    )
    print(f"Output directory: {args.output_dir}")
    print("Created paper-style TPDS, KAT, CPO, and GRFPV figures.")


if __name__ == "__main__":
    main()
