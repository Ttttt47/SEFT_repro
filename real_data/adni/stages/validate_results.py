#!/usr/bin/env python3
"""Validate the ADNI3 reproduction outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PRIMARY_COUNTS = {
    ("CN_gt_Dementia", 0.2): 62,
    ("MCI_gt_Dementia", 0.2): 36,
    ("CN_gt_MCI", 0.1): 25,
}
STABLE_COUNTS = {"CN_gt_Dementia": 42, "MCI_gt_Dementia": 19, "CN_gt_MCI": 10}
OVERLAP_COUNTS = {
    "CN_gt_Dementia": {"shared_discoveries": 42, "seft_only": 20, "simes_bh_only": 8},
    "MCI_gt_Dementia": {"shared_discoveries": 14, "seft_only": 22, "simes_bh_only": 7},
}
FIGURES = (
    "zmap_montage_all.pdf", "co_score_model_check_spatial.pdf",
    "adni_regional_working_models_tfce.pdf", "adni_seft_bh_discovery_counts.pdf",
    "adni_resampling_stability_summary.pdf", "adni_scientific_question_overlap.pdf",
    "zmap_inflated_surface_all.pdf", "zmap_mip_all.pdf", "zmap_model_check_qq_hist.pdf",
)


def good(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def same_geometry(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(left.affine, right.affine, atol=1e-4)


def permutation_accounting_errors(row: dict[str, object], expected_requested: int | None = None) -> list[str]:
    fields = (
        "requested_permutations", "fragments_per_contrast", "permutations_per_fragment",
        "rounded_permutations_per_contrast", "effective_permutation_denominator",
    )
    if any(type(row.get(field)) is not int or row[field] < 1 for field in fields):
        return ["permutation metadata has missing or invalid counts"]
    requested, fragments, per_fragment, rounded, effective = (row[field] for field in fields)
    errors = []
    if expected_requested is not None and requested != expected_requested:
        errors.append(f"requested {requested} permutations, primary analysis used {expected_requested}")
    if rounded != fragments * per_fragment:
        errors.append("rounded permutation count does not match the fragment grid")
    if effective != rounded - fragments + 1:
        errors.append("effective permutation denominator does not match merged fragments")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--strict-reference-counts", action="store_true",
                        help="Also require the discovery counts from the reference run.")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    app = work / "results/application_methods"
    errors: list[str] = []
    checks: dict[str, object] = {}

    required_markers = [
        work / "status/vbm_preprocessing.ok",
        app / "analysis_complete.ok", work / "results/stability_primary/analysis_complete.ok",
    ]
    missing_markers = [str(path) for path in required_markers if not path.is_file()]
    checks["stage_markers"] = {"missing": missing_markers}
    errors.extend(f"missing stage marker: {path}" for path in missing_markers)

    sample_path = work / "provenance/cohort_selection.json"
    sample = json.loads(sample_path.read_text())
    checks["sample"] = sample
    if sample.get("selected_counts") != {"CN": 108, "Dementia": 108, "MCI": 108}:
        errors.append(f"unexpected selected counts: {sample.get('selected_counts')}")

    design = pd.read_csv(app / "tables/design_diagnostics.tsv", sep="\t").iloc[0]
    checks["design"] = design.to_dict()
    if (
        int(design["residual_df"]) != 307
        or int(design["rank"]) != int(design["n_columns"])
    ):
        errors.append("GLM rank or residual degrees of freedom differs from the reference analysis")

    atlas = nib.load(str(Path(__file__).resolve().parents[3] / "real_data/input/atlas/AAL3v1.nii.gz"))
    mask = nib.load(str(work / "fslvbm/stats/GM_mask.nii.gz"))
    if not same_geometry(atlas, mask):
        errors.append("AAL3 atlas and GM mask geometry differ")
    labels = np.rint(np.asarray(atlas.dataobj)).astype(np.int32)
    active = np.asarray(mask.dataobj) > 0
    region_ids = sorted(int(value) for value in np.unique(labels[active]) if value > 0)
    checks["aal3_nonempty_regions"] = len(region_ids)
    if len(region_ids) != 166:
        errors.append(f"expected 166 nonempty AAL3 regions, found {len(region_ids)}")

    summary = pd.read_csv(app / "tables/seft_simes_region_results_long.tsv", sep="\t")
    observed_counts: dict[str, int] = {}
    for (contrast, level), expected in PRIMARY_COUNTS.items():
        block = summary[
            summary.atlas.eq("aal3") & summary.variant.eq("sigma3")
            & summary.contrast.eq(contrast) & np.isclose(summary.pc_level, level)
            & np.isclose(summary.alpha, 0.1) & summary.method.eq("seft")
        ]
        seft_column = "seft_reject" if "seft_reject" in block else "significant"
        observed = int(pd.to_numeric(block[seft_column]).sum())
        observed_counts[f"{contrast}@{level:g}"] = observed
        if args.strict_reference_counts and observed != expected:
            errors.append(f"{contrast} c={level:g}: expected {expected} SEFT discoveries, found {observed}")
    checks["primary_discoveries"] = observed_counts

    stability_path = work / "results/stability_primary/tables/region_stability_summary.tsv"
    stability = pd.read_csv(stability_path, sep="\t")
    stable_counts = stability[
        stability.bootstrap_selection_frequency.ge(0.7)
        & stability.delete10_selection_frequency.ge(0.7)
        & stability.full_sample_selected.eq(1)
    ].groupby("contrast").size().to_dict()
    checks["stable_discoveries"] = stable_counts
    if args.strict_reference_counts and stable_counts != STABLE_COUNTS:
        errors.append(f"unexpected stable-region counts: {stable_counts}")

    overlap = pd.read_csv(app / "qc/tables/seft_simes_overlap_summary.tsv", sep="\t")
    observed_overlap: dict[str, dict[str, int]] = {}
    for contrast, expected in OVERLAP_COUNTS.items():
        row = overlap[overlap.contrast.eq(contrast) & np.isclose(overlap.pc_level, 0.2)]
        if len(row) != 1:
            errors.append(f"missing unique c=0.2 overlap row for {contrast}")
            continue
        observed = {column: int(row.iloc[0][column]) for column in expected}
        observed_overlap[contrast] = observed
        if args.strict_reference_counts and observed != expected:
            errors.append(f"unexpected SEFT/BH overlap for {contrast}: {observed}")
    checks["seft_bh_overlap"] = observed_overlap

    randomise_metadata = sorted((app / "randomise").glob("*_permutation_metadata.json"))
    permutation_rows = [json.loads(path.read_text()) for path in randomise_metadata]
    checks["randomise"] = permutation_rows
    if not permutation_rows:
        errors.append("no randomise permutation metadata found")
    primary_path = work / "provenance/primary_methods.json"
    expected_requested = None
    if primary_path.exists():
        expected_requested = json.loads(primary_path.read_text()).get("n_permutations_requested")
    checks["randomise_requested_by_primary"] = expected_requested
    for path, row in zip(randomise_metadata, permutation_rows, strict=True):
        messages = permutation_accounting_errors(row, expected_requested)
        errors.extend(f"{path.name}: {message}" for message in messages)
    requested_values = {row.get("requested_permutations") for row in permutation_rows}
    if len(requested_values) > 1:
        errors.append("randomise variants have inconsistent requested permutation counts")

    figure_dir = work / "manuscript/figures"
    missing_figures = [name for name in FIGURES if not good(figure_dir / name)]
    checks["paper_figures"] = {"required": list(FIGURES), "missing_or_small": missing_figures}
    errors.extend(f"missing/blank paper figure: {name}" for name in missing_figures)

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work), "passed": not errors, "errors": errors, "checks": checks,
    }
    output = work / "provenance/validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    marker = work / "status/application_complete.ok"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(report["validated_at_utc"] + "\n")
    print(f"PASS: ADNI3 reproduction outputs ({output})")


if __name__ == "__main__":
    main()
