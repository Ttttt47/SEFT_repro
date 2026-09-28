"""Resolve date-stamped ADNI metadata exports by their stable prefixes."""

from __future__ import annotations

from pathlib import Path


METADATA_FILE_PREFIXES = {
    "key_mri": "All_Subjects_Key_MRI",
    "dxsum": "All_Subjects_DXSUM",
    "ptdemog": "All_Subjects_PTDEMOG",
    "study_entry": "All_Subjects_Study_Entry",
}


def resolve_metadata_file(data_dir: str | Path, table: str) -> Path:
    """Return the unique CSV whose name starts with the table's ADNI prefix."""
    data = Path(data_dir).resolve()
    try:
        prefix = METADATA_FILE_PREFIXES[table]
    except KeyError as error:
        raise ValueError(f"Unknown ADNI metadata table: {table}") from error
    matches = sorted(path for path in data.glob(f"{prefix}*.csv") if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No ADNI metadata CSV matching {prefix}*.csv was found under {data}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Expected exactly one ADNI metadata CSV matching {prefix}*.csv under {data}; "
            f"found {len(matches)}: {names}"
        )
    return matches[0]


def resolve_metadata_files(data_dir: str | Path) -> dict[str, Path]:
    """Resolve all metadata inputs required by the ADNI application."""
    return {
        table: resolve_metadata_file(data_dir, table)
        for table in METADATA_FILE_PREFIXES
    }
