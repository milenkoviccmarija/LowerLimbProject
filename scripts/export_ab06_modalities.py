"""Izvozi izabrane AB06 modalitete iz MATLAB fajlova u Parquet.

Ulaz je CSV indeks koji pravi `build_dataset_index.py`.
Za svaku probu skripta cita trazene `.mat` fajlove, dodaje osnovne podatke o
probi i snima tabelu u `data/processed/ab06_modalities`.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matio
import pandas as pd

DEFAULT_INDEX = Path("data/processed/ab06_trial_index.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/ab06_modalities")
DEFAULT_MODALITIES = ["imu", "fp", "conditions", "gcLeft", "gcRight"]


def scalar_text(value: Any) -> str:
    """Pretvara MATLAB liste/celije u kratak tekst."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return scalar_text(value[0])
        return "|".join(scalar_text(item) for item in value)
    return str(value)


def scalar_value(value: Any) -> Any:
    """Pretvara MATLAB scalar vrednosti u obicne Python vrednosti."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return scalar_value(value[0])
        return scalar_text(value)
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return value


def normalize_dataframe(df: pd.DataFrame, *, modality: str, row: pd.Series) -> pd.DataFrame:
    """Ujednacava nazive kolona i dodaje podatke o probi."""
    out = df.copy()

    # MATLAB tabele cesto imaju vreme u koloni `Header`.
    if "Header" in out.columns:
        out = out.rename(columns={"Header": "time_s"})

    # `conditions` tabela nosi labelu kretanja.
    if modality == "conditions" and "Label" in out.columns:
        out = out.rename(columns={"Label": "label"})
        out["label"] = out["label"].map(scalar_text)
    if modality == "conditions" and "label" not in out.columns:
        out["label"] = row["activity_folder"]

    # Ove kolone ostaju uz svaki red da bi se kasnije znalo kojoj probi pripada.
    out.insert(0, "trial_stem", row["trial_stem"])
    out.insert(0, "activity_folder", row["activity_folder"])
    out.insert(0, "session", row["session"])
    out.insert(0, "subject", row["subject"])
    return out


def read_modality_table(path: Path, modality: str) -> pd.DataFrame:
    """Cita jednu MATLAB tabelu za dati modalitet."""
    data = matio.load_from_mat(path, add_table_attrs=True)
    key = "labels" if modality == "conditions" else "data"
    table = data.get(key)

    # Kod `conditions` fajlova podaci ponekad nisu pod istim kljucem.
    if modality == "conditions" and table is None:
        for value in data.values():
            if isinstance(value, pd.DataFrame):
                return value
        meta = {
            name: scalar_value(value)
            for name, value in data.items()
            if not str(name).startswith("__") and not isinstance(value, pd.DataFrame)
        }
        return pd.DataFrame([meta])
    if not isinstance(table, pd.DataFrame):
        available = ", ".join(data.keys())
        raise ValueError(f"Expected DataFrame key {key!r} in {path}; available: {available}")  # noqa: TRY004
    return table


def export_modalities(
    index_csv: Path,
    output_dir: Path,
    modalities: list[str],
    *,
    limit: int | None,
    overwrite: bool,
    skip_static: bool,
) -> pd.DataFrame:
    """Izvozi izabrane modalitete i pravi manifest izvezenih fajlova."""
    index = pd.read_csv(index_csv)

    # Static probe nisu korisne za osnovnu klasifikaciju kretanja.
    if skip_static:
        index = index[index["activity_folder"] != "static"]
    if limit is not None:
        index = index.head(limit)

    manifest_rows: list[dict[str, Any]] = []
    for _, row in index.iterrows():
        activity_dir = output_dir / str(row["activity_folder"])
        activity_dir.mkdir(parents=True, exist_ok=True)

        for modality in modalities:
            source = str(row.get(f"{modality}_path", "") or "")
            if not source:
                continue
            source_path = Path(source)
            target_path = activity_dir / f"{row['trial_stem']}__{modality}.parquet"

            # Ako fajl vec postoji, ne pisi ga opet osim ako korisnik trazi overwrite.
            if target_path.exists() and not overwrite:
                manifest_rows.append(
                    {
                        "subject": row["subject"],
                        "session": row["session"],
                        "activity_folder": row["activity_folder"],
                        "trial_stem": row["trial_stem"],
                        "modality": modality,
                        "source_path": source,
                        "output_path": str(target_path),
                        "rows": "",
                        "columns": "",
                        "status": "exists",
                    }
                )
                continue

            # Citanje `.mat`, normalizacija kolona i snimanje u Parquet.
            table = read_modality_table(source_path, modality)
            table = normalize_dataframe(table, modality=modality, row=row)
            table.attrs = {}
            table.to_parquet(target_path, index=False)
            manifest_rows.append(
                {
                    "subject": row["subject"],
                    "session": row["session"],
                    "activity_folder": row["activity_folder"],
                    "trial_stem": row["trial_stem"],
                    "modality": modality,
                    "source_path": source,
                    "output_path": str(target_path),
                    "rows": len(table),
                    "columns": len(table.columns),
                    "status": "written",
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest je spisak svega sto je izvezeno ili preskoceno.
    manifest.to_csv(output_dir / "export_manifest.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Cita opcije prosledjene kroz komandnu liniju."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-static", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Pokrece export i ispisuje kratak rezultat."""
    args = parse_args()
    if not args.index_csv.exists():
        raise SystemExit(f"Index CSV not found: {args.index_csv}. Run build_dataset_index.py first.")

    manifest = export_modalities(
        args.index_csv,
        args.output_dir,
        args.modalities,
        limit=args.limit,
        overwrite=args.overwrite,
        skip_static=not args.include_static,
    )
    status_counts = manifest["status"].value_counts().to_dict() if not manifest.empty else {}
    print(f"Exported manifest rows: {len(manifest)}")
    print(f"Status counts: {status_counts}")
    print(f"Manifest: {args.output_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
