#!/usr/bin/env python3
"""Run the unified three-group GLM, TFCE reference, SEFT and Simes analyses."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = REPRO_ROOT / "outputs/adni3"
FSLDIR = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))
SEFT_ENV = Path(sys.executable).resolve().parent.parent
TOOLS = REPRO_ROOT / "tools/seft_fsl"
RANDOMISE_PARALLEL = REPRO_ROOT / "real_data/adni/fsl/randomise_parallel_limited.sh"
AAL3 = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
AAL3_LABELS = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.txt"
HO = Path(os.environ.get("SEFT_HO_ATLAS", DEFAULT_WORK / "atlas/harvard_oxford_thr25_2mm.nii.gz"))
HO_LABELS = Path(os.environ.get("SEFT_HO_LABELS", DEFAULT_WORK / "atlas/harvard_oxford_thr25_2mm_labels.tsv"))

sys.path.insert(0, str(TOOLS))
from seft_fsl_lib import read_tsv, run_seft_analysis  # noqa: E402


SEED = 20260717
GROUPS = ["CN", "MCI", "Dementia"]
CONTRASTS = [
    ("CN_gt_Dementia", [1, 0, -1]),
    ("MCI_gt_Dementia", [0, 1, -1]),
    ("CN_gt_MCI", [1, -1, 0]),
    ("Dementia_gt_CN", [-1, 0, 1]),
    ("Dementia_gt_MCI", [0, -1, 1]),
    ("MCI_gt_CN", [-1, 1, 0]),
]
PRIMARY = {"CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI"}
PRIMARY_BANDWIDTH = 5.0
PRIMARY_NEIGHBOR_RANGE = 10


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def env() -> dict[str, str]:
    output = os.environ.copy()
    output["FSLDIR"] = str(FSLDIR)
    output["PATH"] = f"{SEFT_ENV / 'bin'}:{FSLDIR / 'share/fsl/bin'}:{FSLDIR / 'bin'}:" + output.get("PATH", "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "FSL_NUM_THREADS"):
        output[key] = "1"
    return output


def good(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_cmd(cmd: list[object], log_path: Path, command_env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(map(str, cmd)) + "\n")
        handle.flush()
        proc = subprocess.run([str(value) for value in cmd], env=command_env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"[exit] {proc.returncode}\n")
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")


def wait_for(paths: list[Path], timeout_hours: int = 168) -> None:
    start = time.time()
    while time.time() - start < timeout_hours * 3600:
        missing = [path for path in paths if not good(path)]
        if not missing:
            return
        log(f"waiting for {len(missing)} randomise outputs; first={missing[0].name}")
        time.sleep(30)
    raise TimeoutError("Timed out waiting for randomise outputs")


def zscore(values: pd.Series, *, log_transform: bool = False) -> np.ndarray:
    array = pd.to_numeric(values, errors="raise").to_numpy(float)
    if log_transform:
        array = np.log(array)
    return (array - array.mean()) / array.std(ddof=1)


def write_fsl_mat(path: Path, matrix: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(f"/NumWaves\t{matrix.shape[1]}\n/NumPoints\t{matrix.shape[0]}\n")
        handle.write("/PPheights\t" + "\t".join(["1"] * matrix.shape[1]) + "\n/Matrix\n")
        np.savetxt(handle, matrix, fmt="%.10g", delimiter="\t")


def write_fsl_con(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    with path.open("w") as handle:
        for index, name in enumerate(names, start=1):
            handle.write(f"/ContrastName{index}\t{name}\n")
        handle.write(f"/NumWaves\t{matrix.shape[1]}\n/NumContrasts\t{matrix.shape[0]}\n")
        handle.write("/PPheights\t" + "\t".join(["1"] * matrix.shape[0]) + "\n")
        handle.write("/RequiredEffect\t" + "\t".join(["1"] * matrix.shape[0]) + "\n/Matrix\n")
        np.savetxt(handle, matrix, fmt="%.10g", delimiter="\t")


def vif_table(matrix: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        y = matrix[:, index]
        other = np.delete(matrix, index, axis=1)
        fitted = other @ np.linalg.lstsq(other, y, rcond=None)[0]
        total = float(np.sum((y - y.mean()) ** 2))
        residual = float(np.sum((y - fitted) ** 2))
        r2 = 1 - residual / total if total > 0 else np.nan
        rows.append({"column": name, "r_squared_from_other_columns": r2,
                     "vif": 1 / (1 - r2) if np.isfinite(r2) and r2 < 1 else np.nan})
    return pd.DataFrame(rows)


def build_design(rows: pd.DataFrame, tables: Path) -> tuple[Path, Path, int]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for group in GROUPS:
        arrays.append(rows.label.eq(group).to_numpy(float))
        names.append(group)
    for column, name, log_transform in (
        ("age_at_scan_est", "age_z", False),
        ("sex_male", "sex_male_z", False),
        ("pteducat", "education_z", False),
        ("tiv_mm3", "log_tiv_z", True),
    ):
        arrays.append(zscore(rows[column], log_transform=log_transform))
        names.append(name)
    protocol = pd.get_dummies(rows.scanner_protocol_family_raw.astype(str), prefix="scanner_protocol", dtype=float)
    protocol = protocol.reindex(sorted(protocol.columns), axis=1)
    for column in protocol.columns[1:]:
        value = protocol[column].to_numpy(float)
        arrays.append(value - value.mean())
        names.append(column)
    matrix = np.column_stack(arrays)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise RuntimeError(f"Rank-deficient design: rank={rank}, columns={matrix.shape[1]}")
    contrasts = np.zeros((len(CONTRASTS), matrix.shape[1]), dtype=float)
    contrasts[:, :3] = np.asarray([values for _, values in CONTRASTS], dtype=float)
    mat = tables / "design.mat"
    con = tables / "design.con"
    write_fsl_mat(mat, matrix)
    write_fsl_con(con, contrasts, [name for name, _ in CONTRASTS])
    pd.DataFrame(matrix, columns=names).to_csv(tables / "design_matrix.tsv", sep="\t", index=False)
    vif_table(matrix, names).to_csv(tables / "design_vif.tsv", sep="\t", index=False)
    condition = float(np.linalg.cond(matrix))
    pd.DataFrame([{
        "n_subjects": len(rows), "n_columns": matrix.shape[1], "rank": rank,
        "residual_df": len(rows) - rank, "condition_number": condition,
    }]).to_csv(tables / "design_diagnostics.tsv", sep="\t", index=False)
    return mat, con, len(rows) - rank


def standardized_difference(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def write_sample_tables(rows: pd.DataFrame, tables: Path) -> None:
    balance: list[dict[str, object]] = []
    for left, right in (("CN", "Dementia"), ("MCI", "Dementia"), ("CN", "MCI")):
        for variable, transform in (("age_at_scan_est", False), ("sex_male", False), ("pteducat", False), ("tiv_mm3", True)):
            a = pd.to_numeric(rows.loc[rows.label.eq(left), variable]).to_numpy(float)
            b = pd.to_numeric(rows.loc[rows.label.eq(right), variable]).to_numpy(float)
            if transform:
                a, b = np.log(a), np.log(b)
            balance.append({
                "comparison": f"{left}_minus_{right}", "variable": "log_tiv" if transform else variable,
                "left_mean": float(a.mean()), "left_sd": float(a.std(ddof=1)),
                "right_mean": float(b.mean()), "right_sd": float(b.std(ddof=1)),
                "standardized_mean_difference": standardized_difference(a, b),
            })
    pd.DataFrame(balance).to_csv(tables / "covariate_balance.tsv", sep="\t", index=False)

    table1: list[dict[str, object]] = []
    for group in GROUPS:
        block = rows[rows.label.eq(group)]
        table1.extend([
            {"variable": "n", "category": "", "group": group, "value": str(len(block))},
            {"variable": "age, years", "category": "mean (SD)", "group": group,
             "value": f"{block.age_at_scan_est.mean():.2f} ({block.age_at_scan_est.std(ddof=1):.2f})"},
            {"variable": "education, years", "category": "mean (SD)", "group": group,
             "value": f"{block.pteducat.mean():.2f} ({block.pteducat.std(ddof=1):.2f})"},
            {"variable": "TIV, mL", "category": "mean (SD)", "group": group,
             "value": f"{block.tiv_mm3.mean()/1000:.1f} ({block.tiv_mm3.std(ddof=1)/1000:.1f})"},
        ])
        for sex_value, sex_label in ((0, "female"), (1, "male")):
            n_value = int(block.sex_male.eq(sex_value).sum())
            table1.append({"variable": "sex", "category": sex_label, "group": group,
                           "value": f"{n_value} ({100*n_value/len(block):.1f}%)"})
    pd.DataFrame(table1).to_csv(tables / "table1.tsv", sep="\t", index=False)
    pd.crosstab(rows.scanner_protocol_family_raw, rows.label).to_csv(tables / "scanner_protocol_balance.tsv", sep="\t")


def bh(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    pvalues = np.nan_to_num(np.clip(pvalues.astype(float), 0, 1), nan=1.0, posinf=1.0, neginf=1.0)
    order = np.argsort(pvalues)
    passed = pvalues[order] <= alpha * np.arange(1, len(pvalues) + 1) / len(pvalues)
    decision = np.zeros(len(pvalues), dtype=int)
    if passed.any():
        decision[order[: np.where(passed)[0].max() + 1]] = 1
    return decision


def atlas_label_lookup(path: Path) -> dict[int, str]:
    if path.suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
        return dict(zip(frame.region_id.astype(int), frame.label.astype(str)))
    lookup: dict[int, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        values = line.strip().split()
        if len(values) >= 2:
            try:
                lookup[int(float(values[0]))] = values[1]
            except ValueError:
                pass
    return lookup


def main() -> None:
    global FSLDIR, SEFT_ENV, HO, HO_LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--fsldir", type=Path, default=FSLDIR)
    parser.add_argument("--seft-env", type=Path, default=SEFT_ENV)
    parser.add_argument("--ho-atlas", type=Path, default=HO)
    parser.add_argument("--ho-labels", type=Path, default=HO_LABELS)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seft-jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    FSLDIR = args.fsldir.resolve()
    SEFT_ENV = args.seft_env.resolve()
    HO = args.ho_atlas.resolve()
    HO_LABELS = args.ho_labels.resolve()
    work = args.work_dir.resolve()
    log(
        f"using fixed paper-default spatial scale: bandwidth={PRIMARY_BANDWIDTH:g} voxels, "
        f"neighbor_range={PRIMARY_NEIGHBOR_RANGE} voxels"
    )
    out = work / "results/application_methods"
    for name in ("inputs", "randomise", "seft_runs", "zmaps", "tables", "qc", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    command_env = env()
    os.environ.update(command_env)
    command_env["MPLCONFIGDIR"] = str(out / "logs/matplotlib")
    Path(command_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.update(command_env)

    if not good(work / "status/vbm_preprocessing.ok"):
        raise SystemExit("VBM preprocessing is not complete")
    rows = pd.read_csv(work / "tables/analysis_order.tsv", sep="\t")
    expected = np.arange(1, len(rows) + 1)
    if not np.array_equal(rows.four_d_index_1based.to_numpy(), expected):
        raise RuntimeError("Invalid 4D subject order")
    counts = rows.groupby("label").size().to_dict()
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Unbalanced final groups: {counts}")
    rows.to_csv(out / "tables/analysis_subjects.tsv", sep="\t", index=False)
    write_sample_tables(rows, out / "tables")
    design_mat, design_con, residual_df = build_design(rows, out / "tables")

    stats = work / "fslvbm/stats"
    mask = stats / "GM_mask.nii.gz"
    variants = {
        "unsmoothed": stats / "GM_mod_merg.nii.gz",
        "sigma2": stats / "GM_mod_merg_s2.nii.gz",
    }
    sigma3_image = stats / "GM_mod_merg_s3.nii.gz"
    if good(sigma3_image):
        variants["sigma3"] = sigma3_image
    for path in [mask, *variants.values()]:
        if not good(path):
            raise RuntimeError(f"Missing VBM analysis image: {path}")

    output_roots = {variant: out / "randomise" / f"three_group_{variant}" for variant in variants}
    required_randomise = {
        variant: [
            Path(f"{root}_tstat{index}.nii.gz") for index in range(1, len(CONTRASTS) + 1)
        ] + [
            Path(f"{root}_tfce_corrp_tstat{index}.nii.gz") for index in range(1, len(CONTRASTS) + 1)
        ]
        for variant, root in output_roots.items()
    }
    pending = [variant for variant, paths in required_randomise.items() if not all(good(path) for path in paths)]
    if pending:
        # A randomise fragment needs roughly 2 GiB for this 324-subject 4D
        # input.  Keep the *combined* fragment concurrency at 64 even when a
        # larger general worker allowance is supplied; two simultaneous
        # 64-slot launchers exceed this server's usable memory.
        randomise_budget = min(args.threads, 64)
        slots = max(1, randomise_budget // len(pending))
        log(f"launching {len(pending)} TFCE runs with {slots} fragment slots each "
            f"(combined randomise budget={randomise_budget})")

        def launch_randomise(variant: str) -> str:
            run_cmd([
                "bash", RANDOMISE_PARALLEL, "--threads", slots, "--requested-time", "120", "--",
                "-i", variants[variant], "-o", output_roots[variant], "-m", mask,
                "-d", design_mat, "-t", design_con, "-n", args.n_perm, "-T", "-V", "--quiet",
            ], out / "logs" / f"randomise_{variant}_launcher.log", command_env)
            return variant

        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            for future in as_completed([pool.submit(launch_randomise, variant) for variant in pending]):
                log(f"randomise fragments launched: {future.result()}")
        wait_for([path for variant in pending for path in required_randomise[variant]])

    atlas_specs = {
        "aal3": (AAL3, AAL3_LABELS),
        "harvard_oxford": (HO, HO_LABELS),
    }
    # Geometry is checked again inside run_seft_analysis.  This explicit check
    # gives an early, readable error before the parallel R jobs start.
    reference = nib.load(str(next(iter(required_randomise["sigma2"]))))
    for atlas_name, (atlas_path, _) in atlas_specs.items():
        atlas_img = nib.load(str(atlas_path))
        if atlas_img.shape[:3] != reference.shape[:3] or not np.allclose(atlas_img.affine, reference.affine, atol=1e-4):
            raise RuntimeError(f"{atlas_name} atlas geometry does not match VBM statistic maps")

    tasks: list[dict[str, object]] = []
    for index, (contrast, _) in enumerate(CONTRASTS, start=1):
        tasks.append({
            "variant": "sigma2", "atlas_name": "aal3", "contrast_index": index,
            "contrast": contrast, "pc_levels": [0.1, 0.2, 0.3] if contrast in PRIMARY else [0.2],
            "save_internals": contrast in PRIMARY,
            "bandwidth": PRIMARY_BANDWIDTH, "neighbor_range": PRIMARY_NEIGHBOR_RANGE,
        })
    for index, (contrast, _) in enumerate(CONTRASTS[:3], start=1):
        tasks.append({"variant": "unsmoothed", "atlas_name": "aal3", "contrast_index": index,
                      "contrast": contrast, "pc_levels": [0.1, 0.2, 0.3], "save_internals": False,
                      "bandwidth": PRIMARY_BANDWIDTH, "neighbor_range": PRIMARY_NEIGHBOR_RANGE})
        if "sigma3" in variants:
            tasks.append({"variant": "sigma3", "atlas_name": "aal3", "contrast_index": index,
                          "contrast": contrast, "pc_levels": [0.1, 0.2, 0.3], "save_internals": True,
                          "bandwidth": PRIMARY_BANDWIDTH, "neighbor_range": PRIMARY_NEIGHBOR_RANGE})
            tasks.append({"variant": "sigma3", "atlas_name": "harvard_oxford", "contrast_index": index,
                          "contrast": contrast, "pc_levels": [0.1, 0.2, 0.3], "save_internals": False,
                          "bandwidth": PRIMARY_BANDWIDTH, "neighbor_range": PRIMARY_NEIGHBOR_RANGE})
        tasks.append({"variant": "sigma2", "atlas_name": "harvard_oxford", "contrast_index": index,
                      "contrast": contrast, "pc_levels": [0.1, 0.2, 0.3], "save_internals": False,
                      "bandwidth": 5.0, "neighbor_range": 10})

    def seft_task(task: dict[str, object]) -> dict[str, object]:
        variant = str(task["variant"])
        atlas_name = str(task["atlas_name"])
        contrast = str(task["contrast"])
        contrast_index = int(task["contrast_index"])
        prefix = f"{variant}_{contrast}_{atlas_name}"
        run_dir = out / "seft_runs" / prefix
        summary = run_dir / "tables" / f"{prefix}_summary.tsv"
        internal_r = run_dir / "maps/claw_internal" / f"{prefix}_R.nii.gz"
        need_internal = bool(task["save_internals"])
        if not good(summary) or (need_internal and not good(internal_r)):
            atlas_path, labels_path = atlas_specs[atlas_name]
            result = run_seft_analysis(
                zmap=None,
                tstat=Path(f"{output_roots[variant]}_tstat{contrast_index}.nii.gz"),
                design_mat=design_mat,
                atlas=atlas_path,
                atlas_labels=labels_path,
                mask=mask,
                out_dir=run_dir,
                prefix=prefix,
                alpha=0.1,
                pc_levels=list(task["pc_levels"]),
                simes=True,
                seed=args.seed,
                denoise="wavelet",
                bandwidth=float(task.get("bandwidth", 5.0)),
                lambda_value=0.5,
                neighbor_range=int(task.get("neighbor_range", 10)),
                score_clip_c=0.99,
                save_internals=need_internal,
                rscript=str(SEFT_ENV / "bin/Rscript"),
                build_maps=True,
            )
            signed_z = result.signed_z
        else:
            signed_z = run_dir / "maps" / f"{prefix}_signed_z.nii.gz"
        central_z = out / "zmaps" / f"{variant}_{contrast}_signed_z.nii.gz"
        if atlas_name == "aal3" and not good(central_z):
            shutil.copy2(signed_z, central_z)
        return {**task, "prefix": prefix, "run_dir": str(run_dir), "summary": str(summary), "signed_z": str(signed_z)}

    completed_tasks: list[dict[str, object]] = []
    log(f"running {len(tasks)} SEFT/Simes configurations with {min(args.seft_jobs, len(tasks))} jobs")
    with ThreadPoolExecutor(max_workers=min(args.seft_jobs, len(tasks))) as pool:
        futures = [pool.submit(seft_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed_tasks.append(result)
            log(f"SEFT complete: {result['variant']} {result['contrast']} {result['atlas_name']}")
    pd.DataFrame(completed_tasks).sort_values(["atlas_name", "variant", "contrast_index"]).to_csv(
        out / "tables/seft_run_manifest.tsv", sep="\t", index=False
    )

    all_region_rows: list[pd.DataFrame] = []
    for task in completed_tasks:
        prefix = str(task["prefix"])
        path = Path(str(task["run_dir"])) / "tables" / f"{prefix}_region_results.tsv"
        frame = pd.read_csv(path, sep="\t")
        frame.insert(0, "contrast", task["contrast"])
        frame.insert(0, "variant", task["variant"])
        frame.insert(0, "atlas", task["atlas_name"])
        for alpha in (0.05, 0.1):
            alpha_frame = frame.copy()
            alpha_frame["alpha"] = alpha
            alpha_frame["significant_recomputed"] = 0
            for (_, method, pc_level), indices in alpha_frame.groupby(["contrast", "method", "pc_level"]).groups.items():
                alpha_frame.loc[indices, "significant_recomputed"] = bh(
                    alpha_frame.loc[indices, "pc_p_value"].to_numpy(float), alpha
                )
            all_region_rows.append(alpha_frame)
    region_long = pd.concat(all_region_rows, ignore_index=True)
    region_long.to_csv(out / "tables/seft_simes_region_results_long.tsv", sep="\t", index=False)
    summary = region_long.groupby(
        ["atlas", "variant", "contrast", "pc_level", "method", "alpha"], as_index=False
    ).agg(total_regions=("region_id", "size"), discovered_regions=("significant_recomputed", "sum"))
    summary.to_csv(out / "tables/seft_simes_summary.tsv", sep="\t", index=False)
    main_summary = summary[
        summary.atlas.eq("aal3") & summary.variant.eq("sigma3") & summary.pc_level.eq(0.2) & summary.alpha.eq(0.1)
    ].copy()
    main_summary.to_csv(out / "tables/primary_seft_simes_summary.tsv", sep="\t", index=False)
    sensitivity = summary[
        summary.contrast.isin(PRIMARY)
        & (
            (summary.atlas.eq("aal3") & summary.variant.isin(["sigma2", "sigma3", "unsmoothed"]))
            | (summary.atlas.eq("harvard_oxford") & summary.variant.eq("sigma3"))
        )
    ].copy()
    sensitivity.to_csv(out / "tables/seft_sensitivity_summary.tsv", sep="\t", index=False)

    mask_data = np.asarray(nib.load(str(mask)).dataobj) > 0
    aal3_data = np.rint(np.asarray(nib.load(str(AAL3)).dataobj)).astype(int)
    aal3_labels = atlas_label_lookup(AAL3_LABELS)
    tfce_rows: list[dict[str, object]] = []
    tfce_region_rows: list[dict[str, object]] = []
    map_manifest: list[dict[str, object]] = []
    for variant, root in output_roots.items():
        for index, (contrast, _) in enumerate(CONTRASTS, start=1):
            tstat = Path(f"{root}_tstat{index}.nii.gz")
            corrp = Path(f"{root}_tfce_corrp_tstat{index}.nii.gz")
            zmap = out / "zmaps" / f"{variant}_{contrast}_signed_z.nii.gz"
            corrp_data = np.asarray(nib.load(str(corrp)).dataobj, dtype=np.float32)
            values = corrp_data[mask_data & np.isfinite(corrp_data)]
            tfce_rows.append({
                "variant": variant, "contrast": contrast, "n_mask_voxels": int(values.size),
                "significant_voxels_fwer_0p05": int(np.sum(values >= 0.95)),
                "significant_fraction_fwer_0p05": float(np.mean(values >= 0.95)),
                "max_one_minus_p": float(values.max()), "tfce_corrp_map": str(corrp),
            })
            for region_id in sorted(int(value) for value in np.unique(aal3_data[mask_data]) if value > 0):
                region_mask = mask_data & aal3_data.__eq__(region_id)
                region_values = corrp_data[region_mask]
                n_sig = int(np.sum(region_values >= 0.95))
                tfce_region_rows.append({
                    "variant": variant, "contrast": contrast, "region_id": region_id,
                    "region_label": aal3_labels.get(region_id, f"region_{region_id}"),
                    "region_voxels_in_gm_mask": int(region_mask.sum()),
                    "tfce_significant_voxels_fwer_0p05": n_sig,
                    "tfce_coverage_fraction": n_sig / int(region_mask.sum()),
                    "max_one_minus_p": float(region_values.max()) if region_values.size else np.nan,
                })
            for map_type, path in (("tstat", tstat), ("tfce_corrp", corrp), ("signed_z", zmap)):
                if good(path):
                    map_manifest.append({"variant": variant, "contrast": contrast, "map_type": map_type, "path": str(path)})
    pd.DataFrame(tfce_rows).to_csv(out / "tables/tfce_summary.tsv", sep="\t", index=False)
    pd.DataFrame(tfce_region_rows).to_csv(out / "tables/tfce_by_aal3_region.tsv", sep="\t", index=False)
    pd.DataFrame(map_manifest).to_csv(out / "tables/statistical_map_manifest.tsv", sep="\t", index=False)

    metadata = {
        "completed_at": stamp(), "seed": args.seed, "n_permutations_requested": args.n_perm,
        "threads": args.threads, "residual_df": residual_df, "group_counts": counts,
        "contrasts": [name for name, _ in CONTRASTS],
        "primary_directions": sorted(PRIMARY),
        "primary_seft": {
            "atlas": "AAL3", "smoothing_sigma_mm": 3, "denoise": "wavelet",
            "pc_fraction_c": 0.2, "alpha": 0.1,
            "bandwidth_voxels": PRIMARY_BANDWIDTH,
            "kernel_sigma_mm": 2.0 * PRIMARY_BANDWIDTH,
            "neighbor_range_voxels": PRIMARY_NEIGHBOR_RANGE,
            "bandwidth_selection": "fixed paper-default value; not estimated from the observed maps",
        },
        "regional_pc_test_sidedness": "two-sided; contrast sign is summarized separately from the regional decision",
        "score_clip_c": 0.99,
        "tfce_interpretation": "voxelwise FWER reference for spatial corroboration; not a region-level power comparator",
    }
    (work / "provenance/primary_methods.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (out / "analysis_complete.ok").write_text(stamp() + "\n")
    log("Primary methods complete")


if __name__ == "__main__":
    main()
