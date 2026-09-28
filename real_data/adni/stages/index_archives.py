#!/usr/bin/env python3
"""Index approved ADNI ZIP archives without extracting their DICOM payloads."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[3]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from real_data.adni.metadata_files import METADATA_FILE_PREFIXES, resolve_metadata_file


KEY_PREFIX = METADATA_FILE_PREFIXES["key_mri"]
FIELDS = [
    "image_id", "subject_id", "visit", "diagnosis_code", "label", "zip",
    "dicom_files", "series_type", "series_description",
]


def load_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        rows = csv.DictReader(handle)
        required = {"image_id", "subject_id", "image_visit", "image_date", "series_type", "series_description"}
        missing = required.difference(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return {row["image_id"].strip(): row for row in rows if row["image_id"].strip()}


def index_zip(path: Path, key: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    counts: Counter[str] = Counter()
    subjects: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(".dcm"):
                continue
            parts = name.split("/")
            image_match = next((re.fullmatch(r"I(\d+)", part) for part in parts if re.fullmatch(r"I\d+", part)), None)
            if image_match is None:
                continue
            image_id = image_match.group(1)
            counts[image_id] += 1
            if len(parts) >= 3:
                subjects[image_id] = parts[1]
                descriptions[image_id] = parts[2].replace("_", " ")
    output = []
    for image_id, count in sorted(counts.items(), key=lambda item: int(item[0])):
        metadata = key.get(image_id)
        if metadata is None:
            raise ValueError(f"Image I{image_id} from {path.name} is absent from the MRI key table")
        subject = metadata["subject_id"].strip()
        if subjects.get(image_id, subject) != subject:
            raise ValueError(f"Subject mismatch for image I{image_id} in {path.name}")
        output.append({
            "image_id": image_id,
            "subject_id": subject,
            "visit": metadata["image_visit"].strip(),
            "diagnosis_code": "",
            "label": "",
            "zip": path.name,
            "dicom_files": str(count),
            "series_type": metadata["series_type"].strip(),
            "series_description": metadata["series_description"].strip() or descriptions.get(image_id, ""),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--downloads-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    downloads = (args.downloads_dir or data / "downloads").resolve()
    output = (args.output or data / "download_image_stats/image_level_manifest.csv").resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    archives = sorted(downloads.glob("*.zip"))
    if not archives:
        raise SystemExit(f"No ZIP archives found under {downloads}")
    key_path = resolve_metadata_file(data, "key_mri")
    key = load_key(key_path)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(archives))) as pool:
        blocks = list(pool.map(lambda path: index_zip(path, key), archives))
    rows = [row for block in blocks for row in block]
    image_ids = [row["image_id"] for row in rows]
    duplicates = [item for item, count in Counter(image_ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Image IDs occur in multiple archives: {duplicates[:10]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(
        f"Indexed {len(rows)} image series from {len(archives)} archives using "
        f"{key_path.name}: {output}"
    )


if __name__ == "__main__":
    main()
