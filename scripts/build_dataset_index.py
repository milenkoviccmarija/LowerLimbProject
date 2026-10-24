"""Pravi indeks svih AB06 proba.

Skripta prolazi kroz AB06 foldere, pronalazi `.mat` fajlove i za svaku probu
belezi:

- osnovne podatke iz `conditions` fajla
- koje modalitete proba ima
- putanje do pronadjenih fajlova

Rezultat je CSV indeks dataset-a.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.io import loadmat

DEFAULT_DATASET_ROOT = Path("AB06")
DEFAULT_OUTPUT_CSV = Path("data/processed/ab06_trial_index.csv")

# Redosled modaliteta koji se koristi za CSV kolone.
MODALITY_ORDER = [
    "conditions",
    "imu",
    "fp",
    "emg",
    "gcLeft",
    "gcRight",
    "gon",
    "ik",
    "ik_offset",
    "id",
    "jp",
    "markers",
]


def matlab_scalar(value: Any) -> Any:
    """Pretvara MATLAB/SciPy vrednosti u obicne Python vrednosti."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        return matlab_scalar(value[0])
    return value


def read_conditions(path: Path) -> dict[str, Any]:
    """Cita metadata iz jednog `conditions` fajla."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    result: dict[str, Any] = {}
    for key, value in mat.items():
        if key.startswith("__") or key == "None":
            continue
        result[key] = matlab_scalar(value)
    return result


def discover_modalities(session_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Path]], Counter]:
    """Pronalazi `.mat` fajlove u sesiji i grupise ih po probi."""
    trial_map: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    modality_counts: Counter = Counter()

    for mat_file in session_dir.glob("*/*/*.mat"):
        activity = mat_file.parents[1].name
        modality = mat_file.parent.name
        stem = mat_file.stem
        trial_map[(activity, stem)][modality] = mat_file
        modality_counts[(activity, modality)] += 1

    return trial_map, modality_counts


def build_index(dataset_root: Path) -> pd.DataFrame:
    """Pravi tabelu u kojoj je svaki red jedna proba."""
    # Jedna sesija je npr. folder `10_09_18`.
    session_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name != "osimxml")
    rows: list[dict[str, Any]] = []

    for session_dir in session_dirs:
        trial_map, _ = discover_modalities(session_dir)

        for (activity, stem), modalities in sorted(trial_map.items()):
            # Osnovni opis probe i spisak dostupnih modaliteta.
            row: dict[str, Any] = {
                "subject": dataset_root.name,
                "session": session_dir.name,
                "activity_folder": activity,
                "trial_stem": stem,
                "modality_count": len(modalities),
                "modalities_present": ",".join(m for m in MODALITY_ORDER if m in modalities),
                "missing_modalities": ",".join(m for m in MODALITY_ORDER if m not in modalities),
            }

            for modality in MODALITY_ORDER:
                row[f"{modality}_path"] = str(modalities[modality]) if modality in modalities else ""

            # `conditions` sadrzi labelu, brzinu, smer i vreme trajanja probe.
            conditions_path = modalities.get("conditions")
            if conditions_path is not None:
                try:
                    meta = read_conditions(conditions_path)
                    row.update(
                        {
                            "trial_type": meta.get("trialType", ""),
                            "speed": meta.get("speed", ""),
                            "turn": meta.get("turn", ""),
                            "leading_leg_start": meta.get("leadingLegStart", ""),
                            "leading_leg_stop": meta.get("leadingLegStop", ""),
                            "label_error": meta.get("labelError", ""),
                            "trial_start_s": meta.get("trialStarts", ""),
                            "trial_end_s": meta.get("trialEnds", ""),
                            "source_file": meta.get("file", ""),
                        }
                    )
                    if row["trial_start_s"] != "" and row["trial_end_s"] != "":
                        row["duration_s"] = float(row["trial_end_s"]) - float(row["trial_start_s"])
                    else:
                        row["duration_s"] = ""
                except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                    # Nastavi indeksiranje cak i ako je jedan fajl problematican.
                    row["conditions_error"] = repr(exc)
            rows.append(row)

    df = pd.DataFrame(rows)
    preferred_columns = [
        "subject",
        "session",
        "activity_folder",
        "trial_stem",
        "trial_type",
        "speed",
        "turn",
        "leading_leg_start",
        "leading_leg_stop",
        "label_error",
        "trial_start_s",
        "trial_end_s",
        "duration_s",
        "modality_count",
        "modalities_present",
        "missing_modalities",
        "source_file",
    ]
    path_columns = [f"{modality}_path" for modality in MODALITY_ORDER]
    other_columns = [column for column in df.columns if column not in preferred_columns + path_columns]
    return df[[column for column in preferred_columns + path_columns + other_columns if column in df.columns]]


def parse_args() -> argparse.Namespace:
    """Cita opcije prosledjene kroz komandnu liniju."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    """Pokrece indeksiranje i snima izlazne fajlove."""
    args = parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")

    df = build_index(args.dataset_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(df)} indexed trials to {args.output_csv}")


if __name__ == "__main__":
    main()
