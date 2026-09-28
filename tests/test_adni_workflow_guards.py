from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cohort = load("prepare_cohort", ROOT / "real_data/adni/stages/prepare_cohort.py")
pipeline = load("adni_pipeline", ROOT / "real_data/adni/run_pipeline.py")


def test_matching_is_deterministic_and_exact() -> None:
    rows = []
    for group, ages in {
        "CN": [70.0, 80.0], "MCI": [70.2, 79.8], "Dementia": [70.1, 79.9]
    }.items():
        for index, age in enumerate(ages):
            rows.append({
                "vbm_id": f"{group}_{index}", "label": group,
                "age_at_scan_est": age, "pteducat": 16 - index,
                "scanner_protocol_family_raw": "Siemens__MPRAGE", "sex_male": index,
            })
    candidates = pd.DataFrame(rows)
    first = cohort.build_triplets(candidates)
    second = cohort.build_triplets(candidates)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[0].groupby("label").size().to_dict() == {"CN": 2, "Dementia": 2, "MCI": 2}
    assert first[1].scanner_protocol_family_raw.eq("Siemens__MPRAGE").all()


def test_default_pipeline_selects_cohort() -> None:
    assert "cohort" in pipeline.STAGES
    assert "sample" not in pipeline.STAGES
    assert "qc" not in pipeline.STAGES
