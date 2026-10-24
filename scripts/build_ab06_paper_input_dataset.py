"""Build the paper-style input dataset with dataset-specific signal adaptation.

This script creates sample-level rows for the four paper feature groups:

    TPDS  -> theta_t_tn, dot_theta_t_stn, phi_new
    KAT   -> theta_k_sn
    CPO   -> beta_x, beta_y
    GRFPV -> Gtp, Ghp

The input kinematics use thigh/shank accelerometer pitch so `theta_t` and
`theta_s` remain segment angles rather than hip/knee joint-angle proxies. This
version also uses the best available local support signals:

1. Thigh/shank IMU pitch for segment angles.
2. `gcLeft`/`gcRight` gait-cycle phase for cycle validation and normalization.
3. Native force-plate vertical force, reduced to first/second stance peak
   proxies for GRFPV.

`phi_new` follows the paper phase-diagram formula:
atan2(dot_theta_t_stn, theta_t_tn) + pi.

The result is still a local-dataset adaptation, not a direct heel/toe insole
reproduction unless the input dataset provides those exact channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

DEFAULT_MANIFEST = Path("data/processed/ab06_modalities/export_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/paper_input_datasets")
DEFAULT_ACTIVITIES = ["levelground", "treadmill", "ramp", "stair"]
REQUIRED_MODALITIES = ("conditions", "imu", "fp")
OPTIONAL_MODALITIES = ("gcLeft", "gcRight")
PAPER_INPUT_MODES = ("ST", "LW", "RA", "RD", "SA", "SD")
PAPER_STEADY_MODES = {"LW", "RA", "RD", "SA", "SD"}
PAPER_MODE_NAMES = {
    "ST": "standing",
    "LW": "level walking",
    "RA": "ramp ascent",
    "RD": "ramp descent",
    "SA": "stair ascent",
    "SD": "stair descent",
}
PAPER_SCALE_GAMMA = 180.0
PAPER_SMOOTHING_R = 0.9
THIGH_SMOOTHING_WINDOW_S = 0.35
KNEE_SMOOTHING_WINDOW_S = 0.20
DEFAULT_NORMALIZATION_WINDOW_S = 1.0
DEFAULT_IMU_LOWPASS_CUTOFF_HZ = 8.0
DEFAULT_IMU_LOWPASS_ORDER = 2
EPSILON = 1e-9


def paper_mode_label(label: str) -> str:
    """Map AB06 labels to the paper locomotion-mode names."""
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
        return "LW"
    return "unknown"


def pair_manifest(manifest: pd.DataFrame, activities: list[str]) -> pd.DataFrame:
    """Pair exported modality files by trial."""
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

    for modality in (*REQUIRED_MODALITIES, *OPTIONAL_MODALITIES):
        if modality not in paired.columns:
            paired[modality] = pd.NA

    paired = paired.dropna(subset=list(REQUIRED_MODALITIES))
    if paired.empty:
        raise ValueError(
            "No usable trials found. Need conditions, imu, and fp exports for "
            f"at least one of these activities: {activities}"
        )
    return paired


def read_optional_parquet(row: pd.Series, column: str) -> pd.DataFrame | None:
    """Read an optional parquet file from a paired-manifest row."""
    path = row.get(column, pd.NA)
    if pd.isna(path) or not str(path):
        return None
    return pd.read_parquet(path)


def label_imu_rows(imu: pd.DataFrame, conditions: pd.DataFrame, coarse_label: str) -> pd.Series:
    """Assign each IMU sample the nearest condition label."""
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


def numeric_series(values: pd.Series) -> pd.Series:
    """Convert a series to floating-point numeric values."""
    return pd.to_numeric(values, errors="coerce").astype(float)


def align_columns_to_time(
    base_time_s: pd.Series,
    source: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    """Nearest-time align selected source columns to the base time axis."""
    out = pd.DataFrame(index=base_time_s.index)
    for column in columns:
        out[column] = np.nan
    if source is None or "time_s" not in source.columns:
        return out

    present = [column for column in columns if column in source.columns]
    if not present:
        return out

    left = pd.DataFrame({"time_s": base_time_s.to_numpy(dtype=float)}, index=base_time_s.index)
    left_sorted = left.sort_values("time_s")
    right = source[["time_s", *present]].dropna(subset=["time_s"]).sort_values("time_s")
    if right.empty:
        return out

    merged = pd.merge_asof(left_sorted, right, on="time_s", direction="nearest")
    merged.index = left_sorted.index
    merged = merged.reindex(base_time_s.index)
    for column in present:
        out[column] = numeric_series(merged[column])
    return out


def segment_pitch_degrees(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Fallback segment pitch from accelerometer."""
    accel_x = df[f"{prefix}_Accel_X"].to_numpy(dtype=float)
    accel_y = df[f"{prefix}_Accel_Y"].to_numpy(dtype=float)
    accel_z = df[f"{prefix}_Accel_Z"].to_numpy(dtype=float)
    denominator = np.sqrt(np.square(accel_y) + np.square(accel_z))
    return pd.Series(np.degrees(np.arctan2(accel_x, denominator)), index=df.index)


def derivative(values: pd.Series, time_s: pd.Series) -> pd.Series:
    """Numerically differentiate values over time."""
    raw = numeric_series(values).interpolate(limit_direction="both")
    time = numeric_series(time_s).to_numpy(dtype=float)
    signal = raw.to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    if valid.sum() < 3:
        return pd.Series(np.nan, index=values.index, dtype=float)

    safe_time = time.copy()
    if np.any(np.diff(safe_time[valid]) <= 0):
        median_dt = np.median(np.diff(safe_time[valid]))
        if not np.isfinite(median_dt) or median_dt <= 0:
            median_dt = 0.005
        safe_time = np.arange(len(signal), dtype=float) * median_dt

    diff = np.gradient(signal, safe_time)
    return pd.Series(diff, index=values.index, dtype=float)


def choose_kinematics(
    imu: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, str, str, str]:
    """Use IMU segment pitch for thigh and shank angles."""
    time_s = imu["time_s"]

    theta_t = segment_pitch_degrees(imu, "thigh")
    theta_s = segment_pitch_degrees(imu, "shank")
    theta_t_source = "thigh accelerometer pitch"
    theta_k_source = "derived as theta_s minus theta_t from segment pitch"
    dot_theta_t = derivative(theta_t, time_s)
    dot_theta_t_source = f"time derivative of {theta_t_source}"
    return theta_t, dot_theta_t, theta_s, theta_t_source, dot_theta_t_source, theta_k_source


def preferred_gc_order(trial_stem: str) -> list[str]:
    """Choose a gait-cycle table order from the trial stem side hint."""
    lowered = f"_{trial_stem.lower()}_"
    if "_l_" in lowered:
        return ["gcLeft", "gcRight"]
    return ["gcRight", "gcLeft"]


def choose_gait_phase(
    imu: pd.DataFrame,
    gc_left: pd.DataFrame | None,
    gc_right: pd.DataFrame | None,
    trial_stem: str,
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    """Choose a gait-cycle phase signal and derive a cycle id."""
    sources = {"gcLeft": gc_left, "gcRight": gc_right}
    for name in preferred_gc_order(trial_stem):
        aligned = align_columns_to_time(imu["time_s"], sources[name], ["HeelStrike", "ToeOff"])
        phase = aligned["HeelStrike"]
        if phase.notna().mean() < 0.8:
            continue
        wraps = phase.diff().fillna(0.0) < -50.0
        if wraps.sum() < 1:
            continue
        toeoff = aligned["ToeOff"].clip(lower=0.0, upper=100.0)
        cycle_id = wraps.cumsum().astype(int)
        return phase.clip(lower=0.0, upper=100.0), toeoff, cycle_id, f"{name}/HeelStrike"

    phase = pd.Series(np.nan, index=imu.index, dtype=float)
    toeoff = pd.Series(np.nan, index=imu.index, dtype=float)
    cycle_id = pd.Series(np.arange(len(imu)), index=imu.index, dtype=int)
    return phase, toeoff, cycle_id, ""


def force_component_columns(
    fp: pd.DataFrame,
    include_prefixes: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Return numeric force-component columns grouped by axis suffix."""
    grouped: dict[str, list[str]] = {"vx": [], "vy": [], "vz": []}
    for column in fp.columns:
        if column.startswith("Combined_"):
            continue
        if include_prefixes and not column.startswith(include_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(fp[column]):
            continue
        for axis in grouped:  # noqa: PLC0206
            if column.endswith(f"_{axis}"):
                grouped[axis].append(column)
    return grouped


def choose_vertical_force_axis(
    fp: pd.DataFrame,
    include_prefixes: tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    """Choose the dominant force axis instead of assuming `_vz` is vertical."""
    grouped = force_component_columns(fp, include_prefixes=include_prefixes)
    scores = {}
    for axis, columns in grouped.items():
        if not columns:
            scores[axis] = 0.0
            continue
        values = fp[columns].abs().sum(axis=1)
        scores[axis] = float(values.quantile(0.95))

    axis = max(scores, key=scores.get)
    columns = grouped.get(axis, [])
    active = [
        column
        for column in columns
        if float(fp[column].abs().quantile(0.95)) > EPSILON
    ]
    return axis, active


def side_matched_force_prefixes(phase_source: str) -> tuple[str, ...]:
    """Prefer treadmill force from the same side as the selected gait cycle."""
    if phase_source.startswith("gcLeft/"):
        return ("Treadmill_L_",)
    if phase_source.startswith("gcRight/"):
        return ("Treadmill_R_",)
    return ()


def grfpv_force_candidates(
    fp: pd.DataFrame,
    phase_source: str,
) -> tuple[str, list[str], str]:
    """Choose candidate vertical-force columns for stance-peak GRFPV."""
    preferred_prefixes = side_matched_force_prefixes(phase_source)
    if preferred_prefixes:
        axis, columns = choose_vertical_force_axis(fp, include_prefixes=preferred_prefixes)
        if columns:
            return axis, columns, "side-matched treadmill stance peaks"

    axis, columns = choose_vertical_force_axis(fp)
    if columns:
        return axis, columns, "per-cycle dominant force-plate stance peaks"
    return axis, [], "no usable force columns"


def force_contact_thresholds(fp: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Estimate a contact threshold per force source column."""
    thresholds: dict[str, float] = {}
    for column in columns:
        values = numeric_series(fp[column]).abs()
        high = float(values.quantile(0.95))
        if not np.isfinite(high) or high <= EPSILON:
            thresholds[column] = np.inf
        else:
            thresholds[column] = max(20.0, 0.08 * high)
    return thresholds


def force_peak_quality_thresholds(fp: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Estimate a minimum plausible stance peak per force source column."""
    thresholds: dict[str, float] = {}
    for column in columns:
        values = numeric_series(fp[column]).abs()
        high = float(values.quantile(0.95))
        if not np.isfinite(high) or high <= EPSILON:
            thresholds[column] = np.inf
        else:
            thresholds[column] = max(150.0, 0.35 * high)
    return thresholds


def true_regions(mask: np.ndarray, min_samples: int = 3) -> list[np.ndarray]:
    """Return contiguous true regions from a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []

    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        np.arange(start, stop)
        for start, stop in zip(starts, stops)
        if stop - start >= min_samples
    ]


def choose_contact_region(
    force: np.ndarray,
    phase: np.ndarray,
    threshold: float,
) -> np.ndarray | None:
    """Choose the best stance contact region within one gait cycle."""
    if not np.isfinite(threshold):
        return None

    phase_windows = ((0.0, 80.0), (0.0, 100.0), (-np.inf, np.inf))
    for low, high in phase_windows:
        phase_mask = (phase >= low) & (phase <= high)
        contact = np.isfinite(force) & (force >= threshold) & phase_mask
        regions = true_regions(contact)
        if not regions:
            continue
        return max(regions, key=lambda region: float(np.nanmax(force[region])))
    return None


def split_contact_peaks(
    times: np.ndarray,
    force: np.ndarray,
    phase: np.ndarray,
    region: np.ndarray,
) -> dict[str, float]:
    """Extract first and second stance peak values from one contact region."""
    if region.size < 2:
        raise ValueError("contact region must contain at least two samples")

    progress = np.linspace(0.0, 1.0, region.size)
    early = region[progress <= 0.50]
    late = region[progress >= 0.50]
    if early.size == 0:
        early = region[: max(1, region.size // 2)]
    if late.size == 0:
        late = region[max(0, region.size // 2) :]

    early_peak_index = early[int(np.nanargmax(force[early]))]
    late_peak_index = late[int(np.nanargmax(force[late]))]
    return {
        "Ghp": float(force[early_peak_index]),
        "Gtp": float(force[late_peak_index]),
        "contact_start_s": float(times[region[0]]),
        "contact_end_s": float(times[region[-1]]),
        "peak1_time_s": float(times[early_peak_index]),
        "peak2_time_s": float(times[late_peak_index]),
        "peak1_phase_percent": float(phase[early_peak_index]),
        "peak2_phase_percent": float(phase[late_peak_index]),
    }


def empty_grfpv_frame(index: pd.Index, axis: str, columns: list[str], source_type: str) -> pd.DataFrame:
    """Create an empty GRFPV extraction frame aligned to IMU samples."""
    candidate_columns = ",".join(columns)
    return pd.DataFrame(
        {
            "Gh": 0.0,
            "Gt": 0.0,
            "Ghp_fp": np.nan,
            "Gtp_fp": np.nan,
            "grf_vertical_axis": axis,
            "grf_vertical_sources": "",
            "grfpv_candidate_columns": candidate_columns,
            "grf_force_source_type": source_type,
            "grf_contact_threshold": np.nan,
            "grf_contact_valid": False,
            "grfpv_cycle_valid": False,
            "grfpv_source_column": "",
            "grfpv_contact_start_s": np.nan,
            "grfpv_contact_end_s": np.nan,
            "grfpv_peak1_time_s": np.nan,
            "grfpv_peak2_time_s": np.nan,
            "grfpv_peak1_phase_percent": np.nan,
            "grfpv_peak2_phase_percent": np.nan,
        },
        index=index,
    )


def extract_grfpv_stance_peaks(
    imu: pd.DataFrame,
    fp: pd.DataFrame,
    phase_percent: pd.Series,
    cycle_id: pd.Series,
    phase_source: str,
) -> pd.DataFrame:
    """Extract AB06 GRFPV as first/second stance peaks from native FP signals."""
    axis, columns, source_type = grfpv_force_candidates(fp, phase_source)
    out = empty_grfpv_frame(imu.index, axis, columns, source_type)
    if not columns or "time_s" not in fp.columns:
        return out

    thresholds = force_contact_thresholds(fp, columns)
    peak_quality_thresholds = force_peak_quality_thresholds(fp, columns)
    fp_time = numeric_series(fp["time_s"]).to_numpy(dtype=float)
    valid_fp_time = np.isfinite(fp_time)
    if valid_fp_time.sum() < 3:
        return out

    imu_time = numeric_series(imu["time_s"])
    valid_cycle_ids = valid_gait_cycle_ids(phase_percent, cycle_id)
    forces = {column: numeric_series(fp[column]).abs().to_numpy(dtype=float) for column in columns}

    for cycle_key, index in cycle_id.groupby(cycle_id, sort=False).groups.items():
        if int(cycle_key) not in valid_cycle_ids:
            continue

        segment_time = imu_time.loc[index]
        segment_phase = numeric_series(phase_percent.loc[index])
        valid_phase = np.isfinite(segment_time.to_numpy(dtype=float)) & np.isfinite(
            segment_phase.to_numpy(dtype=float)
        )
        if valid_phase.sum() < 3:
            continue

        t_values = segment_time.to_numpy(dtype=float)[valid_phase]
        p_values = segment_phase.to_numpy(dtype=float)[valid_phase]
        order = np.argsort(t_values)
        t_values = t_values[order]
        p_values = p_values[order]
        if np.any(np.diff(t_values) <= 0):
            unique_time, unique_index = np.unique(t_values, return_index=True)
            t_values = unique_time
            p_values = p_values[unique_index]
        if len(t_values) < 3:
            continue

        fp_mask = valid_fp_time & (fp_time >= t_values[0]) & (fp_time <= t_values[-1])
        if fp_mask.sum() < 3:
            continue

        local_times = fp_time[fp_mask]
        local_phase = np.interp(local_times, t_values, p_values)
        best: dict[str, object] | None = None
        for column in columns:
            local_force = forces[column][fp_mask]
            region = choose_contact_region(local_force, local_phase, thresholds[column])
            if region is None:
                continue
            score = float(np.nanmax(local_force[region]))
            if best is None or score > float(best["score"]):
                best = {
                    "column": column,
                    "force": local_force,
                    "region": region,
                    "threshold": thresholds[column],
                    "score": score,
                }

        if best is None:
            continue

        selected_column = str(best["column"])
        selected_force = best["force"]  # type: ignore[assignment]
        selected_region = best["region"]  # type: ignore[assignment]
        threshold = float(best["threshold"])
        peak_values = split_contact_peaks(
            local_times,
            selected_force,  # type: ignore[arg-type]
            local_phase,
            selected_region,  # type: ignore[arg-type]
        )
        if min(peak_values["Ghp"], peak_values["Gtp"]) < peak_quality_thresholds[selected_column]:
            continue

        cycle_index = pd.Index(index)
        cycle_times = imu_time.loc[cycle_index].to_numpy(dtype=float)
        force_on_imu = np.interp(cycle_times, local_times, selected_force)  # type: ignore[arg-type]
        contact_duration = peak_values["contact_end_s"] - peak_values["contact_start_s"]
        if contact_duration <= EPSILON:
            continue

        contact_mask = (
            (cycle_times >= peak_values["contact_start_s"])
            & (cycle_times <= peak_values["contact_end_s"])
            & (force_on_imu >= threshold)
        )
        stance_progress = (cycle_times - peak_values["contact_start_s"]) / contact_duration
        early_mask = contact_mask & (stance_progress <= 0.50)
        late_mask = contact_mask & (stance_progress > 0.50)

        out.loc[cycle_index, "Ghp_fp"] = peak_values["Ghp"]
        out.loc[cycle_index, "Gtp_fp"] = peak_values["Gtp"]
        out.loc[cycle_index, "grf_vertical_sources"] = selected_column
        out.loc[cycle_index, "grf_contact_threshold"] = threshold
        out.loc[cycle_index, "grfpv_cycle_valid"] = True
        out.loc[cycle_index, "grfpv_source_column"] = selected_column
        out.loc[cycle_index, "grfpv_contact_start_s"] = peak_values["contact_start_s"]
        out.loc[cycle_index, "grfpv_contact_end_s"] = peak_values["contact_end_s"]
        out.loc[cycle_index, "grfpv_peak1_time_s"] = peak_values["peak1_time_s"]
        out.loc[cycle_index, "grfpv_peak2_time_s"] = peak_values["peak2_time_s"]
        out.loc[cycle_index, "grfpv_peak1_phase_percent"] = peak_values["peak1_phase_percent"]
        out.loc[cycle_index, "grfpv_peak2_phase_percent"] = peak_values["peak2_phase_percent"]
        out.loc[cycle_index[early_mask], "Gh"] = force_on_imu[early_mask]
        out.loc[cycle_index[late_mask], "Gt"] = force_on_imu[late_mask]
        out.loc[cycle_index[contact_mask], "grf_contact_valid"] = True

    return out


def sampling_hz(time_s: pd.Series, fallback: float = 200.0) -> float:
    """Estimate sample frequency from time."""
    diffs = numeric_series(time_s).diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return fallback
    median_dt = float(diffs.median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        return fallback
    return float(1.0 / median_dt)


def window_samples_from_seconds(time_s: pd.Series, window_s: float, fallback: int) -> int:
    """Convert a time window to samples."""
    fs = sampling_hz(time_s)
    if window_s <= 0 or not np.isfinite(fs):
        return max(2, fallback)
    return max(2, round(window_s * fs))


def lowpass_filter_signal(
    values: pd.Series,
    time_s: pd.Series,
    cutoff_hz: float,
    order: int,
) -> pd.Series:
    """Causal low-pass filtering."""
    raw = numeric_series(values)
    if cutoff_hz <= 0 or order <= 0 or len(raw) < 8:
        return raw

    filled = raw.interpolate(limit_direction="both")
    if filled.isna().all():
        return raw

    fs = sampling_hz(time_s)
    nyquist = 0.5 * fs
    effective_cutoff = min(float(cutoff_hz), 0.95 * nyquist)
    if effective_cutoff <= 0 or not np.isfinite(effective_cutoff):
        return raw

    sos = butter(int(order), effective_cutoff, btype="lowpass", fs=fs, output="sos")
    signal = filled.to_numpy(dtype=float)
    zi = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=zi)
    out = pd.Series(filtered, index=values.index)
    out.loc[raw.isna()] = np.nan
    return out


def linear_fit_smooth(
    values: pd.Series,
    time_s: pd.Series,
    window_samples: int,
    smoothing_r: float = PAPER_SMOOTHING_R,
) -> pd.Series:
    """Smooth a signal with local linear fitting."""
    raw = values.to_numpy(dtype=float)
    time = time_s.to_numpy(dtype=float)
    fitted = np.full(len(raw), np.nan, dtype=float)
    window_samples = max(2, int(window_samples))

    for index in range(len(raw)):
        start = max(0, index - window_samples + 1)
        x = time[start : index + 1]
        y = raw[start : index + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2 or not np.isfinite(time[index]):
            fitted[index] = raw[index]
            continue

        x_valid = x[valid]
        y_valid = y[valid]
        x_mean = x_valid.mean()
        y_mean = y_valid.mean()
        denominator = np.square(x_valid - x_mean).sum()
        if denominator <= EPSILON:
            fitted[index] = y_mean
            continue

        slope = ((x_valid - x_mean) * (y_valid - y_mean)).sum() / denominator
        intercept = y_mean - slope * x_mean
        fitted[index] = slope * time[index] + intercept

    smoothed = (1.0 - smoothing_r) * raw + smoothing_r * fitted
    return pd.Series(smoothed, index=values.index)


def rolling_translate_normalize(
    values: pd.Series,
    window_samples: int,
    gamma: float = PAPER_SCALE_GAMMA,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fallback rolling translation/normalization."""
    window_samples = max(2, int(window_samples))
    min_periods = min(window_samples, max(2, len(values)))
    rolling = values.rolling(window=window_samples, min_periods=min_periods)
    rolling_min = rolling.min()
    rolling_max = rolling.max()
    value_range = rolling_max - rolling_min
    beta = -0.5 * (rolling_max + rolling_min)
    alpha = pd.Series(np.nan, index=values.index, dtype=float)
    valid = value_range.abs() > EPSILON
    alpha.loc[valid] = 1.0 / value_range.loc[valid]
    normalized = gamma * alpha * (values + beta)
    return normalized, alpha, beta


def valid_gait_cycle_ids(
    phase_percent: pd.Series,
    cycle_id: pd.Series,
    min_samples: int = 20,
) -> set[int]:
    """Return cycle ids that look like complete gait cycles."""
    valid_ids: set[int] = set()
    for cycle_key, phase_values in phase_percent.groupby(cycle_id, sort=False):
        clean = numeric_series(phase_values).dropna()
        if len(clean) < min_samples:
            continue
        phase_min = float(clean.min())
        phase_max = float(clean.max())
        if phase_min <= 8.0 and phase_max >= 92.0 and phase_max - phase_min >= 85.0:
            valid_ids.add(int(cycle_key))
    return valid_ids


def cycle_translate_normalize(
    values: pd.Series,
    cycle_id: pd.Series,
    gamma: float = PAPER_SCALE_GAMMA,
    valid_cycle_ids: set[int] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize each true gait cycle independently."""
    normalized = pd.Series(np.nan, index=values.index, dtype=float)
    alpha = pd.Series(np.nan, index=values.index, dtype=float)
    beta = pd.Series(np.nan, index=values.index, dtype=float)

    for cycle_key, cycle_values in values.groupby(cycle_id, sort=False):
        if valid_cycle_ids is not None and int(cycle_key) not in valid_cycle_ids:
            continue
        index = cycle_values.index
        if len(index) < 3:
            continue
        v_min = float(cycle_values.min(skipna=True))
        v_max = float(cycle_values.max(skipna=True))
        value_range = v_max - v_min
        if not np.isfinite(value_range) or abs(value_range) <= EPSILON:
            continue
        cycle_beta = -0.5 * (v_max + v_min)
        cycle_alpha = 1.0 / value_range
        normalized.loc[index] = gamma * cycle_alpha * (cycle_values + cycle_beta)
        alpha.loc[index] = cycle_alpha
        beta.loc[index] = cycle_beta
    return normalized, alpha, beta


def cycle_phase_peaks(
    early_force: pd.Series,
    late_force: pd.Series,
    cycle_id: pd.Series,
    valid_cycle_ids: set[int] | None = None,
    contact_threshold: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Assign early/late vertical-force peaks per gait cycle."""
    ghp = pd.Series(np.nan, index=early_force.index, dtype=float)
    gtp = pd.Series(np.nan, index=late_force.index, dtype=float)

    for cycle_key, index in cycle_id.groupby(cycle_id, sort=False).groups.items():
        if valid_cycle_ids is not None and int(cycle_key) not in valid_cycle_ids:
            continue
        if len(index) < 3:
            continue
        early_values = early_force.loc[index]
        late_values = late_force.loc[index]
        if np.isfinite(contact_threshold) and contact_threshold > 0:
            early_values = early_values[early_values >= contact_threshold]
            late_values = late_values[late_values >= contact_threshold]
        if early_values.empty or late_values.empty:
            continue
        early_peak = float(early_values.max(skipna=True))
        late_peak = float(late_values.max(skipna=True))
        ghp.loc[index] = early_peak if np.isfinite(early_peak) else np.nan
        gtp.loc[index] = late_peak if np.isfinite(late_peak) else np.nan
    return ghp, gtp


def paper_phase_from_thigh_phase_diagram(
    theta_t_tn: pd.Series,
    dot_theta_t_stn: pd.Series,
) -> pd.Series:
    """Compute the paper phase variable from the normalized thigh phase diagram."""
    return pd.Series(
        np.arctan2(dot_theta_t_stn, theta_t_tn) + np.pi,
        index=theta_t_tn.index,
    )


def add_processed_paper_parameters(
    dataset: pd.DataFrame,
    normalization_window_s: float,
    imu_lowpass_cutoff_hz: float,
    imu_lowpass_order: int,
) -> pd.DataFrame:
    """Add translated/normalized paper-style parameters."""
    out = dataset.copy()
    processed_columns = [
        "theta_t_tn",
        "dot_theta_t_f",
        "dot_theta_t_s",
        "dot_theta_t_stn",
        "phi_new",
        "theta_k",
        "theta_k_s",
        "theta_k_sn",
        "beta_x",
        "beta_y",
        "Ghp",
        "Gtp",
        "normalization_window_samples",
        "thigh_smoothing_window_samples",
        "knee_smoothing_window_samples",
        "imu_lowpass_cutoff_hz",
        "imu_lowpass_order",
        "grfpv_contact_threshold",
    ]
    for column in processed_columns:
        out[column] = np.nan
    out["normalization_method"] = ""
    out["valid_gait_cycle"] = False

    segment_id = out["omega_m"].ne(out["omega_m"].shift()).cumsum()
    for _, segment in out.groupby(segment_id, sort=False):
        if len(segment) < 3:
            continue

        index = segment.index
        time_s = segment["time_s"]
        normalization_window = window_samples_from_seconds(
            time_s,
            window_s=normalization_window_s,
            fallback=100,
        )
        thigh_smoothing_window = window_samples_from_seconds(
            time_s,
            window_s=THIGH_SMOOTHING_WINDOW_S,
            fallback=70,
        )
        knee_smoothing_window = window_samples_from_seconds(
            time_s,
            window_s=KNEE_SMOOTHING_WINDOW_S,
            fallback=40,
        )

        theta_t = numeric_series(segment["theta_t"])
        dot_theta_t = numeric_series(segment["dot_theta_t"])
        theta_k = numeric_series(segment["theta_s"]) - theta_t
        dot_theta_t_f = lowpass_filter_signal(
            dot_theta_t,
            time_s,
            cutoff_hz=imu_lowpass_cutoff_hz,
            order=imu_lowpass_order,
        )
        dot_theta_t_s = linear_fit_smooth(dot_theta_t_f, time_s, thigh_smoothing_window)
        theta_k_s = linear_fit_smooth(theta_k, time_s, knee_smoothing_window)
        dynamic_trend = dot_theta_t_s.abs().rolling(
            normalization_window,
            min_periods=min(2, len(dot_theta_t_s)),
        ).mean()

        phase = numeric_series(segment["gait_phase_percent"])
        cycle_id = segment["gait_cycle_id"].astype(int)
        valid_cycle_ids = valid_gait_cycle_ids(phase, cycle_id)
        phase_available = phase.notna().mean() > 0.8
        has_gait_cycles = phase_available and len(valid_cycle_ids) >= 1
        threshold_values = numeric_series(segment["grf_contact_threshold"]).dropna()
        contact_threshold = float(threshold_values.median()) if not threshold_values.empty else 0.0
        mode = str(segment["omega_m"].iloc[0])
        valid_cycle_mask = cycle_id.isin(valid_cycle_ids)

        if mode == "ST":
            theta_t_tn = pd.Series(np.nan, index=index, dtype=float)
            dot_theta_t_stn = pd.Series(np.nan, index=index, dtype=float)
            theta_k_sn = pd.Series(np.nan, index=index, dtype=float)
            phi_new = pd.Series(np.nan, index=index, dtype=float)
            beta_x = pd.Series(np.nan, index=index, dtype=float)
            beta_y = pd.Series(np.nan, index=index, dtype=float)
            if {"Ghp_fp", "Gtp_fp"}.issubset(segment.columns):
                ghp = numeric_series(segment["Ghp_fp"])
                gtp = numeric_series(segment["Gtp_fp"])
            else:
                ghp = pd.Series(np.nan, index=index, dtype=float)
                gtp = pd.Series(np.nan, index=index, dtype=float)
            if ghp.notna().sum() == 0:
                ghp = segment["Gh"].rolling(normalization_window, min_periods=2).max()
            if gtp.notna().sum() == 0:
                gtp = segment["Gt"].rolling(normalization_window, min_periods=2).max()
            normalization_method = "standing classifier: dynamic trend plus GRFPV only"
            valid_cycle_mask = pd.Series(True, index=index)
        elif has_gait_cycles:
            theta_t_tn, _, beta_x = cycle_translate_normalize(
                theta_t,
                cycle_id,
                valid_cycle_ids=valid_cycle_ids,
            )
            dot_theta_t_stn, _, beta_y = cycle_translate_normalize(
                dot_theta_t_s,
                cycle_id,
                valid_cycle_ids=valid_cycle_ids,
            )
            theta_k_sn, _, _ = cycle_translate_normalize(
                theta_k_s,
                cycle_id,
                valid_cycle_ids=valid_cycle_ids,
            )
            phi_new = paper_phase_from_thigh_phase_diagram(theta_t_tn, dot_theta_t_stn)
            if {"Ghp_fp", "Gtp_fp"}.issubset(segment.columns):
                ghp = numeric_series(segment["Ghp_fp"]).where(valid_cycle_mask)
                gtp = numeric_series(segment["Gtp_fp"]).where(valid_cycle_mask)
                if mode == "ST" and (ghp.notna().sum() == 0 or gtp.notna().sum() == 0):
                    ghp, gtp = cycle_phase_peaks(
                        segment["Gh"],
                        segment["Gt"],
                        cycle_id,
                        valid_cycle_ids=valid_cycle_ids,
                        contact_threshold=contact_threshold,
                    )
            else:
                ghp, gtp = cycle_phase_peaks(
                    segment["Gh"],
                    segment["Gt"],
                    cycle_id,
                    valid_cycle_ids=valid_cycle_ids,
                    contact_threshold=contact_threshold,
                )
            normalization_method = "gait-cycle-wise using gc phase"
        else:
            if mode != "ST":
                out.loc[index, "normalization_method"] = (
                    "skipped missing or partial gait-cycle phase"
                )
                out.loc[index, "grfpv_contact_threshold"] = contact_threshold
                continue
            theta_t_tn, _, beta_x = rolling_translate_normalize(theta_t, normalization_window)
            dot_theta_t_stn, _, beta_y = rolling_translate_normalize(dot_theta_t_s, normalization_window)
            theta_k_sn, _, _ = rolling_translate_normalize(theta_k_s, normalization_window)
            phi_new = paper_phase_from_thigh_phase_diagram(theta_t_tn, dot_theta_t_stn)
            ghp = segment["Gh"].rolling(normalization_window, min_periods=2).max()
            gtp = segment["Gt"].rolling(normalization_window, min_periods=2).max()
            normalization_method = "rolling fallback"
            valid_cycle_mask = pd.Series(True, index=index)

        out.loc[index, "theta_t_tn"] = theta_t_tn
        out.loc[index, "dot_theta_t_f"] = dot_theta_t_f
        out.loc[index, "dot_theta_t_s"] = dot_theta_t_s
        out.loc[index, "dot_theta_t_stn"] = dot_theta_t_stn
        out.loc[index, "phi_new"] = phi_new
        out.loc[index, "theta_k"] = theta_k
        out.loc[index, "theta_k_s"] = theta_k_s
        out.loc[index, "theta_k_sn"] = theta_k_sn
        out.loc[index, "beta_x"] = beta_x
        out.loc[index, "beta_y"] = beta_y
        out.loc[index, "Ghp"] = ghp
        out.loc[index, "Gtp"] = gtp
        out.loc[index, "dynamic_trend"] = dynamic_trend
        out.loc[index, "normalization_window_samples"] = normalization_window
        out.loc[index, "thigh_smoothing_window_samples"] = thigh_smoothing_window
        out.loc[index, "knee_smoothing_window_samples"] = knee_smoothing_window
        out.loc[index, "imu_lowpass_cutoff_hz"] = imu_lowpass_cutoff_hz
        out.loc[index, "imu_lowpass_order"] = imu_lowpass_order
        out.loc[index, "normalization_method"] = normalization_method
        out.loc[index, "valid_gait_cycle"] = valid_cycle_mask
        out.loc[index, "grfpv_contact_threshold"] = contact_threshold

    out["processed_note"] = (
        "Paper Eq. 6-10 AB06 adaptation v3: theta_t/theta_s are thigh/shank "
        "IMU segment pitch, paper atan2 thigh phase variable, validated "
        "gait-cycle normalization, native force-plate first/second stance GRF "
        "peak proxies for GRFPV, and a separate ST dynamic-trend path because "
        "ST has no stable phase variable in the paper."
    )
    return out


def rows_for_trial(
    row: pd.Series,
    include_modes: set[str],
    normalization_window_s: float,
    imu_lowpass_cutoff_hz: float,
    imu_lowpass_order: int,
) -> pd.DataFrame:
    """Create paper-style rows for one trial."""
    imu = pd.read_parquet(row["imu"])
    fp = pd.read_parquet(row["fp"])
    conditions = pd.read_parquet(row["conditions"])
    gc_left = read_optional_parquet(row, "gcLeft")
    gc_right = read_optional_parquet(row, "gcRight")

    if "time_s" not in imu.columns:
        raise ValueError(f"{row['trial_stem']} IMU file is missing time_s")

    theta_t, dot_theta_t, theta_s, theta_t_source, dot_theta_t_source, theta_k_source = choose_kinematics(imu)
    phase_percent, toeoff_percent, cycle_id, phase_source = choose_gait_phase(
        imu,
        gc_left,
        gc_right,
        row["trial_stem"],
    )
    grfpv = extract_grfpv_stance_peaks(
        imu=imu,
        fp=fp,
        phase_percent=phase_percent,
        cycle_id=cycle_id,
        phase_source=phase_source,
    )

    labels = label_imu_rows(imu, conditions, row["activity_folder"])
    omega = labels.map(paper_mode_label).astype("string")

    out = pd.DataFrame(
        {
            "subject": row["subject"],
            "session": row["session"],
            "activity_folder": row["activity_folder"],
            "trial_stem": row["trial_stem"],
            "time_s": imu["time_s"].to_numpy(dtype=float),
            "label": labels.astype(str).to_numpy(),
            "omega_m": omega.astype(str).to_numpy(),
            "paper_mode_name": omega.map(PAPER_MODE_NAMES).fillna("").astype(str).to_numpy(),
            "theta_t": theta_t.to_numpy(dtype=float),
            "dot_theta_t": dot_theta_t.to_numpy(dtype=float),
            "theta_s": theta_s.to_numpy(dtype=float),
            "Gh": grfpv["Gh"].to_numpy(dtype=float),
            "Gt": grfpv["Gt"].to_numpy(dtype=float),
            "Ghp_fp": grfpv["Ghp_fp"].to_numpy(dtype=float),
            "Gtp_fp": grfpv["Gtp_fp"].to_numpy(dtype=float),
            "gait_phase_percent": phase_percent.to_numpy(dtype=float),
            "gait_toeoff_percent": toeoff_percent.to_numpy(dtype=float),
            "gait_cycle_id": cycle_id.to_numpy(dtype=int),
            "theta_t_source": theta_t_source,
            "dot_theta_t_source": dot_theta_t_source,
            "theta_s_source": "shank accelerometer pitch",
            "theta_k_source": theta_k_source,
            "phase_source": phase_source,
            "Gh_source": "native force-plate first-half stance force",
            "Gt_source": "native force-plate second-half stance force",
            "grf_vertical_axis": grfpv["grf_vertical_axis"].astype(str).to_numpy(),
            "grf_vertical_sources": grfpv["grf_vertical_sources"].astype(str).to_numpy(),
            "grfpv_candidate_columns": grfpv["grfpv_candidate_columns"].astype(str).to_numpy(),
            "grf_force_source_type": grfpv["grf_force_source_type"].astype(str).to_numpy(),
            "grf_contact_threshold": grfpv["grf_contact_threshold"].to_numpy(dtype=float),
            "grf_contact_valid": grfpv["grf_contact_valid"].to_numpy(dtype=bool),
            "grfpv_cycle_valid": grfpv["grfpv_cycle_valid"].to_numpy(dtype=bool),
            "grfpv_source_column": grfpv["grfpv_source_column"].astype(str).to_numpy(),
            "grfpv_contact_start_s": grfpv["grfpv_contact_start_s"].to_numpy(dtype=float),
            "grfpv_contact_end_s": grfpv["grfpv_contact_end_s"].to_numpy(dtype=float),
            "grfpv_peak1_time_s": grfpv["grfpv_peak1_time_s"].to_numpy(dtype=float),
            "grfpv_peak2_time_s": grfpv["grfpv_peak2_time_s"].to_numpy(dtype=float),
            "grfpv_peak1_phase_percent": grfpv["grfpv_peak1_phase_percent"].to_numpy(dtype=float),
            "grfpv_peak2_phase_percent": grfpv["grfpv_peak2_phase_percent"].to_numpy(dtype=float),
            "grf_note": "Local first/second stance vGRF peak proxy from native force signals, not wearable heel/toe GRF unless those channels are present",
        }
    )
    out = add_processed_paper_parameters(
        out,
        normalization_window_s=normalization_window_s,
        imu_lowpass_cutoff_hz=imu_lowpass_cutoff_hz,
        imu_lowpass_order=imu_lowpass_order,
    )
    return out[out["omega_m"].isin(include_modes)].copy()


def write_by_mode(dataset: pd.DataFrame, output_dir: Path) -> None:
    """Write a separate CSV and Parquet file for each paper mode."""
    by_mode_dir = output_dir / "by_mode"
    by_mode_dir.mkdir(parents=True, exist_ok=True)
    for omega_m, subset in dataset.groupby("omega_m", sort=True):
        name = f"D_raw_{omega_m}"
        subset.to_parquet(by_mode_dir / f"{name}.parquet", index=False)
        subset.to_csv(by_mode_dir / f"{name}.csv", index=False)


def write_summary(dataset: pd.DataFrame, output_dir: Path) -> None:
    """Write a short dataset summary."""
    summary = (
        dataset.groupby("omega_m", sort=True)
        .agg(
            rows=("omega_m", "size"),
            trials=("trial_stem", "nunique"),
            labels=("label", lambda values: ",".join(sorted(set(map(str, values))))),
            theta_t_sources=("theta_t_source", lambda values: ",".join(sorted(set(map(str, values))))),
            theta_k_sources=("theta_k_source", lambda values: ",".join(sorted(set(map(str, values))))),
            phase_sources=("phase_source", lambda values: ",".join(sorted({v for v in map(str, values) if v}))),
            grf_axes=("grf_vertical_axis", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    summary.insert(1, "paper_formula", "D_raw^omega_m = {(theta_t,dot_theta_t,theta_s,Gh,Gt) | omega_m}")
    summary.insert(
        2,
        "processed_formula",
        "D^omega_m = {(theta_t_tn,dot_theta_t_stn,phi_new,theta_k_sn,beta_x,beta_y,Gtp,Ghp) | omega_m}",
    )
    summary.to_csv(output_dir / "summary.csv", index=False)


def build_dataset(
    manifest_path: Path,
    output_dir: Path,
    activities: list[str],
    include_modes: set[str],
    by_mode: bool,
    normalization_window_s: float,
    imu_lowpass_cutoff_hz: float,
    imu_lowpass_order: int,
    limit_trials: int | None,
) -> pd.DataFrame:
    """Build and save the complete paper-style input dataset."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    paired = pair_manifest(manifest, activities)
    if limit_trials is not None:
        paired = paired.head(limit_trials)

    trial_tables = [
        rows_for_trial(
            row,
            include_modes=include_modes,
            normalization_window_s=normalization_window_s,
            imu_lowpass_cutoff_hz=imu_lowpass_cutoff_hz,
            imu_lowpass_order=imu_lowpass_order,
        )
        for _, row in paired.iterrows()
    ]
    dataset = pd.concat(trial_tables, ignore_index=True)
    if dataset.empty:
        raise ValueError(f"No rows were created for modes: {sorted(include_modes)}")

    steady_feature_columns = [
        "theta_t",
        "dot_theta_t",
        "dot_theta_t_f",
        "theta_s",
        "Gh",
        "Gt",
        "theta_t_tn",
        "dot_theta_t_stn",
        "phi_new",
        "theta_k_sn",
        "Ghp",
        "Gtp",
    ]
    standing_feature_columns = [
        "theta_t",
        "dot_theta_t",
        "dot_theta_t_f",
        "theta_s",
        "Gh",
        "Gt",
        "Ghp",
        "Gtp",
        "dynamic_trend",
    ]
    steady = dataset[dataset["omega_m"].isin(PAPER_STEADY_MODES)].dropna(
        subset=steady_feature_columns
    )
    standing = dataset[dataset["omega_m"].eq("ST")].dropna(subset=standing_feature_columns)
    dataset = pd.concat([standing, steady], ignore_index=True)
    if dataset.empty:
        raise ValueError("All rows were removed by paper feature validity filters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "paper_input_samples.parquet", index=False)
    dataset.to_csv(output_dir / "paper_input_samples.csv", index=False)
    write_summary(dataset, output_dir)
    if by_mode:
        write_by_mode(dataset, output_dir)
    return dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activities", nargs="+", default=DEFAULT_ACTIVITIES)
    parser.add_argument("--include-modes", nargs="+", default=list(PAPER_INPUT_MODES))
    parser.add_argument("--normalization-window-s", type=float, default=DEFAULT_NORMALIZATION_WINDOW_S)
    parser.add_argument("--imu-lowpass-cutoff-hz", type=float, default=DEFAULT_IMU_LOWPASS_CUTOFF_HZ)
    parser.add_argument("--imu-lowpass-order", type=int, default=DEFAULT_IMU_LOWPASS_ORDER)
    parser.add_argument("--limit-trials", type=int, default=None)
    parser.add_argument("--no-by-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the dataset build from CLI arguments."""
    args = parse_args()
    dataset = build_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        activities=args.activities,
        include_modes=set(args.include_modes),
        by_mode=not args.no_by_mode,
        normalization_window_s=args.normalization_window_s,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_order=args.imu_lowpass_order,
        limit_trials=args.limit_trials,
    )
    print(f"Output directory: {args.output_dir}")
    print(f"paper_input_samples: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(dataset["omega_m"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
