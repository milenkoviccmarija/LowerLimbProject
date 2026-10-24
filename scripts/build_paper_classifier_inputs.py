"""Build classifier-ready inputs from the paper-style feature tables.

The Machines 2023 paper does not feed raw windows directly into the steady-mode
classifier. It first turns TPDS, KAT, CPO, and GRFPV into mode probabilities,
then uses those values as classifier inputs. This script builds that missing
bridge for the processed project data:

- TPDS: 2D Gaussian per mode and phase bin.
- KAT: 1D Gaussian per mode and phase bin.
- CPO: 2D Gaussian per mode.
- GRFPV: LW line geometry and non-LW circle-like distance geometry.
- ST: separate dynamic-trend plus GRFPV table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PAPER_MODES = ["LW", "SA", "SD", "RA", "RD"]
DEFAULT_STEADY_FEATURES = Path("data/processed/paper_features/paper_steady_features_like_paper.csv")
DEFAULT_STANDING_FEATURES = Path("data/processed/paper_features/paper_standing_features_like_paper.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/paper_classifier_inputs")
EPSILON = 1e-9


@dataclass(frozen=True)
class Gaussian2D:
    mean: np.ndarray
    inv_cov: np.ndarray
    norm: float
    cov: np.ndarray
    count: int


@dataclass(frozen=True)
class Gaussian1D:
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class LineGeometry:
    slope: float
    intercept: float
    x_min: float
    x_max: float


def regularized_cov(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return covariance, inverse covariance, and Gaussian normalizer."""
    cov = np.cov(values, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        cov = np.eye(2)
    scale = max(float(np.trace(cov)) / 2.0, 1.0)
    cov = cov + np.eye(2) * max(EPSILON, scale * 1e-6)
    det = float(np.linalg.det(cov))
    if not np.isfinite(det) or det <= EPSILON:
        cov = cov + np.eye(2) * scale * 1e-3
        det = float(np.linalg.det(cov))
    inv_cov = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(max(det, EPSILON)))
    return cov, inv_cov, norm


def fit_gaussian_2d(values: pd.DataFrame) -> Gaussian2D | None:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return None
    cov, inv_cov, norm = regularized_cov(clean)
    return Gaussian2D(
        mean=clean.mean(axis=0),
        inv_cov=inv_cov,
        norm=norm,
        cov=cov,
        count=len(clean),
    )


def fit_gaussian_1d(values: pd.Series) -> Gaussian1D | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size < 3:
        return None
    std = float(clean.std(ddof=1))
    if not np.isfinite(std) or std <= EPSILON:
        std = 1.0
    return Gaussian1D(mean=float(clean.mean()), std=std, count=int(clean.size))


def pdf_2d(model: Gaussian2D | None, x: float, y: float) -> float:
    if model is None or not np.isfinite([x, y]).all():
        return 0.0
    delta = np.array([x, y], dtype=float) - model.mean
    exponent = -0.5 * float(delta.T @ model.inv_cov @ delta)
    return float(model.norm * np.exp(max(exponent, -700.0)))


def pdf_1d(model: Gaussian1D | None, x: float) -> float:
    if model is None or not np.isfinite(x):
        return 0.0
    z = (float(x) - model.mean) / model.std
    return float(np.exp(max(-0.5 * z * z, -700.0)) / (model.std * np.sqrt(2.0 * np.pi)))


def normalized(scores: dict[str, float]) -> dict[str, float]:
    total = float(sum(value for value in scores.values() if np.isfinite(value) and value > 0.0))
    if total <= EPSILON:
        return {mode: 1.0 / len(scores) for mode in scores}
    return {mode: max(float(scores[mode]), 0.0) / total for mode in scores}


def fit_phase_models(
    df: pd.DataFrame,
    min_count: int,
) -> tuple[dict[tuple[str, int], Gaussian2D], dict[tuple[str, int], Gaussian1D], pd.DataFrame]:
    """Fit TPDS and KAT models by mode and phase bin."""
    tpds_models: dict[tuple[str, int], Gaussian2D] = {}
    kat_models: dict[tuple[str, int], Gaussian1D] = {}
    rows: list[dict[str, Any]] = []

    grouped = df.groupby(["omega_m", "tpds_phi_bin"], sort=True)
    for (mode, phase_bin), subset in grouped:
        if mode not in PAPER_MODES or len(subset) < min_count:
            continue
        phase_key = (str(mode), int(phase_bin))
        tpds = fit_gaussian_2d(subset[["tpds_theta_t_tn", "tpds_dot_theta_t_stn"]])
        kat = fit_gaussian_1d(subset["kat_theta_k_sn"])
        if tpds is not None:
            tpds_models[phase_key] = tpds
        if kat is not None:
            kat_models[phase_key] = kat
        rows.append(
            {
                "omega_m": mode,
                "tpds_phi_bin": int(phase_bin),
                "count": len(subset),
                "tpds_theta_mean": np.nan if tpds is None else tpds.mean[0],
                "tpds_dot_theta_mean": np.nan if tpds is None else tpds.mean[1],
                "tpds_cov_xx": np.nan if tpds is None else tpds.cov[0, 0],
                "tpds_cov_xy": np.nan if tpds is None else tpds.cov[0, 1],
                "tpds_cov_yy": np.nan if tpds is None else tpds.cov[1, 1],
                "kat_mean": np.nan if kat is None else kat.mean,
                "kat_std": np.nan if kat is None else kat.std,
            }
        )

    return tpds_models, kat_models, pd.DataFrame(rows)


def fit_cpo_models(df: pd.DataFrame, min_count: int) -> tuple[dict[str, Gaussian2D], pd.DataFrame]:
    models: dict[str, Gaussian2D] = {}
    rows: list[dict[str, Any]] = []
    for mode, subset in df.groupby("omega_m", sort=True):
        if mode not in PAPER_MODES or len(subset) < min_count:
            continue
        model = fit_gaussian_2d(subset[["cpo_beta_x", "cpo_beta_y"]])
        if model is None:
            continue
        models[str(mode)] = model
        rows.append(
            {
                "omega_m": mode,
                "count": len(subset),
                "cpo_beta_x_mean": model.mean[0],
                "cpo_beta_y_mean": model.mean[1],
                "cpo_cov_xx": model.cov[0, 0],
                "cpo_cov_xy": model.cov[0, 1],
                "cpo_cov_yy": model.cov[1, 1],
            }
        )
    return models, pd.DataFrame(rows)


def fit_lw_line(lw: pd.DataFrame) -> LineGeometry | None:
    clean = lw[["grfpv_Ghp", "grfpv_Gtp"]].dropna()
    if len(clean) < 2:
        return None
    x = clean["grfpv_Ghp"].to_numpy(dtype=float)
    y = clean["grfpv_Gtp"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    return LineGeometry(
        slope=float(slope),
        intercept=float(intercept),
        x_min=float(np.nanmin(x)),
        x_max=float(np.nanmax(x)),
    )


def distance_to_lw_line(x: float, y: float, line: LineGeometry | None) -> float:
    if line is None or not np.isfinite([x, y]).all():
        return np.inf
    x_clamped = min(max(x, line.x_min), line.x_max)
    y_on_line = line.slope * x_clamped + line.intercept
    return float(np.hypot(x - x_clamped, y - y_on_line))


def fit_grfpv_geometry(df: pd.DataFrame) -> tuple[LineGeometry | None, dict[str, tuple[float, float, float]], pd.DataFrame]:
    nonzero = df[(df["grfpv_Ghp"] > EPSILON) & (df["grfpv_Gtp"] > EPSILON)].copy()
    line = fit_lw_line(nonzero[nonzero["omega_m"] == "LW"])
    circles: dict[str, tuple[float, float, float]] = {}
    rows: list[dict[str, Any]] = []
    if line is not None:
        rows.append(
            {
                "omega_m": "LW",
                "geometry": "least_squares_line_segment",
                "count": int((nonzero["omega_m"] == "LW").sum()),
                "line_slope": line.slope,
                "line_intercept": line.intercept,
                "line_x_min": line.x_min,
                "line_x_max": line.x_max,
                "circle_center_x": np.nan,
                "circle_center_y": np.nan,
                "circle_radius_95": np.nan,
            }
        )
    for mode in [mode for mode in PAPER_MODES if mode != "LW"]:
        subset = nonzero[nonzero["omega_m"] == mode]
        if subset.empty:
            continue
        cx = float(subset["grfpv_Ghp"].median())
        cy = float(subset["grfpv_Gtp"].median())
        distances = np.hypot(subset["grfpv_Ghp"] - cx, subset["grfpv_Gtp"] - cy)
        radius = float(distances.quantile(0.95))
        circles[mode] = (cx, cy, radius)
        rows.append(
            {
                "omega_m": mode,
                "geometry": "median_center_95_percent_radius",
                "count": len(subset),
                "line_slope": np.nan,
                "line_intercept": np.nan,
                "line_x_min": np.nan,
                "line_x_max": np.nan,
                "circle_center_x": cx,
                "circle_center_y": cy,
                "circle_radius_95": radius,
            }
        )
    return line, circles, pd.DataFrame(rows)


def grfpv_score(mode: str, ghp: float, gtp: float, line: LineGeometry | None, circles: dict[str, tuple[float, float, float]]) -> float:
    if not np.isfinite([ghp, gtp]).all():
        return 0.0
    if mode == "LW":
        distance = distance_to_lw_line(ghp, gtp, line)
    else:
        circle = circles.get(mode)
        if circle is None:
            return 0.0
        cx, cy, _ = circle
        distance = float(np.hypot(ghp - cx, gtp - cy))
    if not np.isfinite(distance):
        return 0.0
    return float(3.0 / (1.0 + 2.0 * np.exp(min(distance / 100.0, 50.0))))


def build_ann_inputs(df: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create ANN-ready probability columns for the five steady modes."""
    df = df[df["omega_m"].isin(PAPER_MODES)].dropna(
        subset=[
            "tpds_theta_t_tn",
            "tpds_dot_theta_t_stn",
            "tpds_phi_bin",
            "kat_theta_k_sn",
            "cpo_beta_x",
            "cpo_beta_y",
            "grfpv_Gtp",
            "grfpv_Ghp",
        ]
    ).copy()
    if df.empty:
        raise ValueError("No steady rows with complete paper features were found.")

    tpds_models, kat_models, phase_params = fit_phase_models(df, min_count=min_count)
    cpo_models, cpo_params = fit_cpo_models(df, min_count=min_count)
    line, circles, grfpv_params = fit_grfpv_geometry(df)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        phase_bin = int(row["tpds_phi_bin"])
        tpds_scores = {
            mode: pdf_2d(
                tpds_models.get((mode, phase_bin)),
                float(row["tpds_theta_t_tn"]),
                float(row["tpds_dot_theta_t_stn"]),
            )
            for mode in PAPER_MODES
        }
        kat_scores = {
            mode: pdf_1d(kat_models.get((mode, phase_bin)), float(row["kat_theta_k_sn"]))
            for mode in PAPER_MODES
        }
        cpo_scores = {
            mode: pdf_2d(cpo_models.get(mode), float(row["cpo_beta_x"]), float(row["cpo_beta_y"]))
            for mode in PAPER_MODES
        }
        grfpv_scores = {
            mode: grfpv_score(mode, float(row["grfpv_Ghp"]), float(row["grfpv_Gtp"]), line, circles)
            for mode in PAPER_MODES
        }

        tpds_prob = normalized(tpds_scores)
        kat_prob = normalized(kat_scores)
        cpo_prob = normalized(cpo_scores)
        grfpv_prob = normalized(grfpv_scores)

        out = {
            "subject": row["subject"],
            "session": row["session"],
            "activity_folder": row["activity_folder"],
            "trial_stem": row["trial_stem"],
            "time_s": row["time_s"],
            "label": row["label"],
            "omega_m": row["omega_m"],
            "paper_mode_name": row["paper_mode_name"],
            "tpds_phi_bin": phase_bin,
        }
        mean_probs = {}
        for mode in PAPER_MODES:
            out[f"prob_tpds_{mode}"] = tpds_prob[mode]
            out[f"prob_kat_{mode}"] = kat_prob[mode]
            out[f"prob_cpo_{mode}"] = cpo_prob[mode]
            out[f"prob_grfpv_{mode}"] = grfpv_prob[mode]
            mean_probs[mode] = float(
                np.mean([tpds_prob[mode], kat_prob[mode], cpo_prob[mode], grfpv_prob[mode]])
            )
            out[f"prob_mean_{mode}"] = mean_probs[mode]
        out["mean_probability_prediction"] = max(mean_probs, key=mean_probs.get)
        rows.append(out)

    return pd.DataFrame(rows), {
        "tpds_kat_phase_gaussian_parameters": phase_params,
        "cpo_gaussian_parameters": cpo_params,
        "grfpv_geometry_parameters": grfpv_params,
    }


def build_standing_inputs(path: Path) -> pd.DataFrame:
    """Build the separate ST-classifier input table."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    required = ["st_dynamic_trend", "grfpv_Gtp", "grfpv_Ghp"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Standing feature table is missing columns: {sorted(missing)}")
    out = df.dropna(subset=required).copy()
    if out.empty:
        return out
    threshold = float(out["st_dynamic_trend"].quantile(0.95))
    out["st_dynamic_threshold"] = threshold
    out["st_dynamic_candidate"] = out["st_dynamic_trend"] <= threshold
    out["st_feature_note"] = (
        "ST classifier input from Machines 2023: dynamic trend threshold plus GRFPV; TPDS/KAT/CPO are not used for ST."
    )
    return out


def build_classifier_inputs(
    steady_features: Path,
    standing_features: Path,
    output_dir: Path,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_count < 3:
        raise ValueError("min_count must be at least 3 for Gaussian fitting.")
    if not steady_features.exists():
        raise FileNotFoundError(f"Steady feature table not found: {steady_features}")

    steady = pd.read_csv(steady_features)
    ann_inputs, parameter_tables = build_ann_inputs(steady, min_count=min_count)
    standing_inputs = build_standing_inputs(standing_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    ann_inputs.to_csv(output_dir / "paper_ann_classifier_inputs.csv", index=False)
    ann_inputs.to_parquet(output_dir / "paper_ann_classifier_inputs.parquet", index=False)
    if not standing_inputs.empty:
        standing_inputs.to_csv(output_dir / "paper_standing_classifier_inputs.csv", index=False)
        standing_inputs.to_parquet(output_dir / "paper_standing_classifier_inputs.parquet", index=False)
    for name, table in parameter_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return ann_inputs, standing_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady-features", type=Path, default=DEFAULT_STEADY_FEATURES)
    parser.add_argument("--standing-features", type=Path, default=DEFAULT_STANDING_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_inputs, standing_inputs = build_classifier_inputs(
        steady_features=args.steady_features,
        standing_features=args.standing_features,
        output_dir=args.output_dir,
        min_count=args.min_count,
    )
    print(f"Output directory: {args.output_dir}")
    print(f"paper_ann_classifier_inputs: {ann_inputs.shape[0]} rows x {ann_inputs.shape[1]} columns")
    print(
        "paper_standing_classifier_inputs: "
        f"{standing_inputs.shape[0]} rows x {standing_inputs.shape[1]} columns"
    )
    print(ann_inputs["omega_m"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
