#!/usr/bin/env python3
"""Build the frozen ADNI3 CN/MCI/clinical-dementia application cohort.

Selection is intentionally simple: earliest ADNI3 3T T1 visit with an exact
visit-level diagnosis, preferring the standard acquisition over ND/ORIG/REPEAT
variants.  Dementia participants anchor exact sex/scanner/protocol triplets;
CN and MCI matches minimize age and education distance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPRO_ROOT = Path(__file__).resolve().parents[3]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from real_data.adni.metadata_files import resolve_metadata_files


DATA = REPRO_ROOT / "real_data/adni/input"
DEFAULT_WORK = REPRO_ROOT / "outputs/adni3"
SEED = 20260717


def norm_token(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return text or "Unknown"


def protocol_family(description: object) -> str:
    text = str(description or "").upper()
    if "MPRAGE" in text or "MP-RAGE" in text:
        return "MPRAGE"
    if "IR-FSPGR" in text or "IR FSPGR" in text or "IR_FSPGR" in text:
        return "IR_FSPGR"
    return norm_token(description)


def variant_flags(description: object) -> tuple[int, int, int]:
    text = str(description or "").upper()
    nd = int(bool(re.search(r"(?:^|[_\s])ND(?:$|[_\s])", text)))
    orig = int("ORIG" in text)
    repeat = int("REPEAT" in text)
    return nd, orig, repeat


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def smd(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").dropna().to_numpy(float)
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    metadata_paths = resolve_metadata_files(DATA)
    images = pd.read_csv(DATA / "download_image_stats/image_level_manifest.csv", dtype=str)
    key = pd.read_csv(metadata_paths["key_mri"], dtype=str)
    dx = pd.read_csv(metadata_paths["dxsum"], dtype=str)
    demog = pd.read_csv(metadata_paths["ptdemog"], dtype=str)
    entry = pd.read_csv(metadata_paths["study_entry"], dtype=str)

    rows = images[images["series_type"].eq("T1w")].merge(
        key, on="image_id", how="inner", suffixes=("", "_key")
    )
    rows = rows[
        rows["mri_protocol_phase"].eq("ADNI3")
        & pd.to_numeric(rows["magnetic_field_strength"], errors="coerce").eq(3.0)
    ].copy()
    rows["image_date_dt"] = pd.to_datetime(rows["image_date"], errors="coerce")
    rows["image_id_num"] = pd.to_numeric(rows["image_id"], errors="coerce")

    dx = dx[dx["DIAGNOSIS"].isin(["1", "2", "3"])].copy()
    dx["dx_visit"] = dx["VISCODE2"].where(dx["VISCODE2"].fillna("").ne(""), dx["VISCODE"])
    dx["dx_date"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx.sort_values(["PTID", "dx_visit", "dx_date", "ID"]).drop_duplicates(
        ["PTID", "dx_visit"], keep="last"
    )
    rows = rows.merge(
        dx[["PTID", "dx_visit", "EXAMDATE", "DIAGNOSIS", "DXDDUE", "DXMDUE", "SITEID"]],
        left_on=["subject_id", "visit"],
        right_on=["PTID", "dx_visit"],
        how="inner",
    )
    rows["label"] = rows["DIAGNOSIS"].map({"1": "CN", "2": "MCI", "3": "Dementia"})

    # Sex and education are stable subject-level descriptors.  Prefer the same
    # visit, then take the first non-missing subject record.
    demog["demog_visit"] = demog["VISCODE2"].where(
        demog["VISCODE2"].fillna("").ne(""), demog["VISCODE"]
    )
    demog["education_num"] = pd.to_numeric(demog["PTEDUCAT"], errors="coerce")
    demog = demog[demog["PTGENDER"].isin(["1", "2"]) & demog["education_num"].ge(0)].copy()
    exact_demog = demog.sort_values(["PTID", "demog_visit", "ID"]).drop_duplicates(
        ["PTID", "demog_visit"], keep="last"
    )
    subject_demog = demog.sort_values(["PTID", "ID"]).drop_duplicates("PTID", keep="first")
    rows = rows.merge(
        exact_demog[["PTID", "demog_visit", "PTGENDER", "education_num"]],
        left_on=["subject_id", "visit"],
        right_on=["PTID", "demog_visit"],
        how="left",
        suffixes=("", "_exact_demog"),
    )
    rows = rows.merge(
        subject_demog[["PTID", "PTGENDER", "education_num"]],
        left_on="subject_id",
        right_on="PTID",
        how="left",
        suffixes=("_exact", "_subject"),
    )
    rows["ptgender"] = rows["PTGENDER_exact"].fillna(rows["PTGENDER_subject"])
    rows["pteducat"] = rows["education_num_exact"].fillna(rows["education_num_subject"])
    rows["sex_male"] = rows["ptgender"].map({"1": 1, "2": 0})

    entry["entry_age_num"] = pd.to_numeric(entry["entry_age"], errors="coerce")
    entry["entry_date_dt"] = pd.to_datetime(entry["entry_date"], errors="coerce")
    entry = entry.sort_values(["subject_id", "entry_date_dt"]).drop_duplicates("subject_id")
    rows = rows.merge(
        entry[["subject_id", "entry_age_num", "entry_date", "entry_date_dt"]],
        on="subject_id",
        how="left",
    )
    rows["age_at_scan_est"] = rows["entry_age_num"] + (
        rows["image_date_dt"] - rows["entry_date_dt"]
    ).dt.days / 365.25

    flags = rows["series_description"].map(variant_flags)
    rows[["is_nd", "is_orig", "is_repeat"]] = pd.DataFrame(flags.tolist(), index=rows.index)
    rows["variant_priority"] = 100 * rows["is_nd"] + 10 * rows["is_orig"] + 5 * rows["is_repeat"]
    rows["protocol_family"] = rows["series_description"].map(protocol_family)
    rows["scanner_family_raw"] = (
        rows["scanner_manufacturer"].map(norm_token) + "__" + rows["scanner_model"].map(norm_token)
    )
    rows["scanner_protocol_family_raw"] = rows["scanner_family_raw"] + "__" + rows["protocol_family"]
    rows["site_id"] = rows["subject_id"].str.slice(0, 3)
    rows["zip_file"] = rows["zip"]
    rows["zip_path"] = rows["zip"].map(lambda value: str(DATA / "downloads" / value))

    rows = rows[
        rows["image_date_dt"].notna()
        & rows["age_at_scan_est"].notna()
        & rows["sex_male"].notna()
        & rows["pteducat"].notna()
    ].copy()

    # Prefer a standard series within a visit; if none exists, the visit is
    # excluded instead of silently selecting ND/ORIG/REPEAT.
    visit_has_standard = rows.groupby(["subject_id", "visit"])["variant_priority"].transform("min").eq(0)
    excluded_variant_visits = rows.loc[~visit_has_standard, ["subject_id", "visit", "image_date", "series_description"]]
    rows = rows[visit_has_standard & rows["variant_priority"].eq(0)].copy()
    rows = rows.sort_values(["subject_id", "image_date_dt", "image_id_num"])
    rows = rows.drop_duplicates(["subject_id", "visit"], keep="first")
    rows = rows.drop_duplicates("subject_id", keep="first")
    rows["vbm_id"] = rows.apply(
        lambda row: (
            f"adni3_{'DEM' if row.label == 'Dementia' else row.label}_"
            f"{row.site_id}_{row.subject_id}_{row.visit}_I{row.image_id}"
        ),
        axis=1,
    )
    return (
        rows.reset_index(drop=True),
        excluded_variant_visits.reset_index(drop=True),
        metadata_paths,
    )


def standardized_cost(cases: pd.DataFrame, controls: pd.DataFrame, all_rows: pd.DataFrame) -> np.ndarray:
    variables = ["age_at_scan_est", "pteducat"]
    scales = {
        key: float(pd.to_numeric(all_rows[key], errors="coerce").std(ddof=1)) or 1.0
        for key in variables
    }
    cost = np.zeros((len(cases), len(controls)), dtype=float)
    for key in variables:
        a = pd.to_numeric(cases[key], errors="raise").to_numpy(float)[:, None]
        b = pd.to_numeric(controls[key], errors="raise").to_numpy(float)[None, :]
        cost += ((a - b) / scales[key]) ** 2
    return cost


def match_group(cases: pd.DataFrame, controls: pd.DataFrame, all_rows: pd.DataFrame) -> dict[int, int]:
    if cases.empty or controls.empty:
        return {}
    cost = standardized_cost(cases, controls, all_rows)
    case_idx, control_idx = linear_sum_assignment(cost)
    return {int(cases.index[i]): int(controls.index[j]) for i, j in zip(case_idx, control_idx, strict=True)}


def build_triplets(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    candidates = candidates.copy()
    candidates["tie_break"] = rng.random(len(candidates))
    strata = ["scanner_protocol_family_raw", "sex_male"]
    triplets: list[dict[str, object]] = []
    selected_indices: set[int] = set()

    for stratum, block in candidates.groupby(strata, sort=True):
        dem = block[block["label"].eq("Dementia")].sort_values(["age_at_scan_est", "pteducat", "tie_break"])
        mci = block[block["label"].eq("MCI")].sort_values(["age_at_scan_est", "pteducat", "tie_break"])
        cn = block[block["label"].eq("CN")].sort_values(["age_at_scan_est", "pteducat", "tie_break"])
        n_triplets = min(len(dem), len(mci), len(cn))
        if n_triplets == 0:
            continue

        # All but at most a very small number of dementia rows are supportable
        # in the downloaded cohort.  If controls are limiting, first retain the
        # dementia rows with the best MCI match, then match CN to that set.
        mci_map_all = match_group(dem, mci, block)
        if len(mci_map_all) < n_triplets:
            continue
        if len(dem) > n_triplets:
            pair_costs = []
            cost_mci = standardized_cost(dem, mci, block)
            dem_positions = {idx: pos for pos, idx in enumerate(dem.index)}
            mci_positions = {idx: pos for pos, idx in enumerate(mci.index)}
            for dem_idx, mci_idx in mci_map_all.items():
                pair_costs.append((cost_mci[dem_positions[dem_idx], mci_positions[mci_idx]], dem_idx))
            keep_dem_indices = {idx for _, idx in sorted(pair_costs)[:n_triplets]}
            dem = dem.loc[sorted(keep_dem_indices)]
        mci_map = match_group(dem, mci, block)
        cn_map = match_group(dem, cn, block)
        common_dem = sorted(set(mci_map) & set(cn_map))[:n_triplets]
        for dem_idx in common_dem:
            mci_idx, cn_idx = mci_map[dem_idx], cn_map[dem_idx]
            selected_indices.update([dem_idx, mci_idx, cn_idx])
            triplets.append(
                {
                    "triplet_id": "",  # assigned after global deterministic sorting
                    "dementia_index": dem_idx,
                    "mci_index": mci_idx,
                    "cn_index": cn_idx,
                    "scanner_protocol_family_raw": stratum[0],
                    "sex_male": int(float(stratum[1])),
                }
            )

    triplet_df = pd.DataFrame(triplets)
    if triplet_df.empty:
        raise RuntimeError("No complete CN/MCI/Dementia triplets could be formed")
    triplet_df = triplet_df.sort_values(["scanner_protocol_family_raw", "sex_male", "dementia_index"]).reset_index(drop=True)
    triplet_df["triplet_id"] = [f"T{i:03d}" for i in range(1, len(triplet_df) + 1)]
    id_lookup: dict[int, str] = {}
    for row in triplet_df.itertuples(index=False):
        for idx in (row.dementia_index, row.mci_index, row.cn_index):
            id_lookup[int(idx)] = row.triplet_id

    selected = candidates.loc[sorted(selected_indices)].copy()
    selected["triplet_id"] = selected.index.map(id_lookup)
    selected["analysis_role"] = selected["label"]
    selected = selected.sort_values(["triplet_id", "analysis_role"]).reset_index(drop=True)

    pair_rows = []
    for row in triplet_df.itertuples(index=False):
        dem = candidates.loc[row.dementia_index]
        mci = candidates.loc[row.mci_index]
        cn = candidates.loc[row.cn_index]
        pair_rows.append(
            {
                "triplet_id": row.triplet_id,
                "cn_vbm_id": cn.vbm_id,
                "mci_vbm_id": mci.vbm_id,
                "dementia_vbm_id": dem.vbm_id,
                "cn_age": cn.age_at_scan_est,
                "mci_age": mci.age_at_scan_est,
                "dementia_age": dem.age_at_scan_est,
                "cn_education": cn.pteducat,
                "mci_education": mci.pteducat,
                "dementia_education": dem.pteducat,
                "scanner_protocol_family_raw": row.scanner_protocol_family_raw,
                "sex_male": row.sex_male,
            }
        )
    return selected, pd.DataFrame(pair_rows), candidates.loc[~candidates.index.isin(selected_indices)].copy()


def balance_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in ["age_at_scan_est", "pteducat"]:
        for left, right in [("CN", "Dementia"), ("MCI", "Dementia"), ("CN", "MCI")]:
            a = selected.loc[selected["label"].eq(left), variable]
            b = selected.loc[selected["label"].eq(right), variable]
            rows.append(
                {
                    "variable": variable,
                    "comparison": f"{left}_minus_{right}",
                    "left_mean": float(pd.to_numeric(a).mean()),
                    "left_sd": float(pd.to_numeric(a).std(ddof=1)),
                    "right_mean": float(pd.to_numeric(b).mean()),
                    "right_sd": float(pd.to_numeric(b).std(ddof=1)),
                    "standardized_mean_difference": smd(a, b),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    work = args.work_dir.resolve()
    tables = work / "tables"
    provenance = work / "provenance"
    tables.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    candidates, excluded_variant_visits, metadata_paths = load_candidates()
    selected, triplets, unused = build_triplets(candidates)

    keep_columns = [
        "triplet_id", "analysis_role", "vbm_id", "label", "subject_id", "site_id",
        "visit", "image_date", "image_id", "zip_file", "zip_path", "series_description",
        "scanner_manufacturer", "scanner_model", "software_version", "magnetic_field_strength",
        "protocol_family", "scanner_family_raw", "scanner_protocol_family_raw", "age_at_scan_est",
        "ptgender", "sex_male", "pteducat", "DIAGNOSIS", "EXAMDATE", "DXDDUE", "DXMDUE",
    ]
    selected_out = selected[keep_columns].rename(columns={"visit": "image_visit", "DIAGNOSIS": "diagnosis_code"})
    write_tsv(tables / "sample_plan.tsv", selected_out)
    write_tsv(tables / "matched_triplets.tsv", triplets)
    write_tsv(tables / "covariate_balance_pre_tiv.tsv", balance_table(selected))
    write_tsv(tables / "eligible_one_scan_candidates.tsv", candidates[keep_columns[2:]])
    write_tsv(tables / "unused_eligible_candidates.tsv", unused[keep_columns[2:]])
    write_tsv(tables / "excluded_no_standard_variant_visits.tsv", excluded_variant_visits)

    counts = selected_out.groupby("label").size().to_dict()
    metadata = {
        "seed": SEED,
        "selection": "earliest exact-diagnosis ADNI3 3T T1 standard variant",
        "matching": "exact sex/scanner/protocol; minimum age/education distance",
        "eligible_counts": candidates.groupby("label").size().to_dict(),
        "selected_counts": counts,
        "n_triplets": int(len(triplets)),
        "n_unique_subjects": int(selected_out["subject_id"].nunique()),
        "source_tables": ["download_image_stats/image_level_manifest.csv"] + [
            path.name for path in metadata_paths.values()
        ],
    }
    (provenance / "cohort_selection.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
