from __future__ import annotations

from pathlib import Path

import pytest

from real_data.adni.metadata_files import resolve_metadata_file, resolve_metadata_files


def test_resolves_date_suffixed_metadata_exports(tmp_path: Path) -> None:
    expected = {
        "key_mri": "All_Subjects_Key_MRI_15Aug2026.csv",
        "dxsum": "All_Subjects_DXSUM_14Aug2026.csv",
        "ptdemog": "All_Subjects_PTDEMOG_13Aug2026.csv",
        "study_entry": "All_Subjects_Study_Entry_12Aug2026.csv",
    }
    for name in expected.values():
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    resolved = resolve_metadata_files(tmp_path)
    assert {table: path.name for table, path in resolved.items()} == expected


def test_rejects_multiple_snapshots_for_one_prefix(tmp_path: Path) -> None:
    (tmp_path / "All_Subjects_DXSUM_01Aug2026.csv").touch()
    (tmp_path / "All_Subjects_DXSUM_15Aug2026.csv").touch()
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        resolve_metadata_file(tmp_path, "dxsum")


def test_reports_missing_metadata_prefix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"All_Subjects_Key_MRI\*\.csv"):
        resolve_metadata_file(tmp_path, "key_mri")
