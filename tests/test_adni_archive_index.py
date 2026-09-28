from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "real_data/adni/stages/index_archives.py"
spec = importlib.util.spec_from_file_location("index_archives", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_zip_index_uses_key_metadata(tmp_path: Path) -> None:
    key_path = tmp_path / f"{module.KEY_PREFIX}_fixture.csv"
    fields = ["image_id", "subject_id", "image_visit", "image_date", "series_type", "series_description"]
    with key_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"image_id": "123", "subject_id": "002_S_0001", "image_visit": "bl",
                         "image_date": "2020-01-02", "series_type": "T1w", "series_description": "MPRAGE"})
    archive = tmp_path / "ADNI3_fixture.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for index in range(3):
            handle.writestr(f"ADNI/002_S_0001/MPRAGE/date/I123/{index}.dcm", b"DICOM")
    rows = module.index_zip(archive, module.load_key(key_path))
    assert rows == [{"image_id": "123", "subject_id": "002_S_0001", "visit": "bl",
                     "diagnosis_code": "", "label": "", "zip": archive.name,
                     "dicom_files": "3", "series_type": "T1w", "series_description": "MPRAGE"}]


def test_index_stage_is_idempotent(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    key_path = tmp_path / f"{module.KEY_PREFIX}_15Aug2026.csv"
    fields = ["image_id", "subject_id", "image_visit", "image_date", "series_type", "series_description"]
    with key_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"image_id": "7", "subject_id": "002_S_0007", "image_visit": "m12",
                         "image_date": "2022-02-03", "series_type": "T1w", "series_description": "MPRAGE"})
    with zipfile.ZipFile(downloads / "ADNI3_fixture.zip", "w") as handle:
        handle.writestr("ADNI/002_S_0007/MPRAGE/date/I7/one.dcm", b"DICOM")
    command = [sys.executable, str(SCRIPT), "--data-dir", str(tmp_path), "--workers", "2"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output = tmp_path / "download_image_stats/image_level_manifest.csv"
    first = output.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_bytes() == first
    assert not (output.parent / f".{output.name}.tmp").exists()
