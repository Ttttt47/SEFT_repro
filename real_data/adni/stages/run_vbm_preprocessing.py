#!/usr/bin/env python3
"""Run the frozen 3-group ADNI3 BET/FAST/FSL-VBM preprocessing.

The implementation follows the FSL-VBM sequence but parallelises independent
subject operations explicitly.  It is restartable: a subject job is skipped
when its expected output already exists, and every stage has a status table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = REPRO_ROOT / "outputs/adni3"
DEFAULT_FSLDIR = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def configure_env(fsldir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FSLDIR"] = str(fsldir)
    env["PATH"] = f"{fsldir}/share/fsl/bin:{fsldir}/bin:" + env.get("PATH", "")
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "GOTO_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "FSL_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def run_cmd(cmd: list[object], log_path: Path, *, cwd: Path | None, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(str(x) for x in cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            [str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
        handle.write(f"[exit] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(str(x) for x in cmd)}")


def good(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def image_stats(path: Path) -> dict[str, float | int | str]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    finite = np.isfinite(data)
    nonzero = finite & (data != 0)
    zooms = image.header.get_zooms()[:3]
    return {
        "shape": "x".join(map(str, data.shape[:3])),
        "voxel_volume_mm3": float(np.prod(zooms)),
        "nonzero_voxels": int(nonzero.sum()),
        "nonzero_volume_mm3": float(nonzero.sum() * np.prod(zooms)),
        "mean_nonzero": float(data[nonzero].mean()) if nonzero.any() else np.nan,
        "min": float(data[finite].min()) if finite.any() else np.nan,
        "max": float(data[finite].max()) if finite.any() else np.nan,
    }


def run_parallel(
    name: str,
    rows: pd.DataFrame,
    worker,
    workers: int,
    status_path: Path,
) -> pd.DataFrame:
    log(f"{name}: {len(rows)} subjects, {min(workers, len(rows))} parallel jobs")
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        futures = {pool.submit(worker, row): row.vbm_id for row in rows.itertuples(index=False)}
        for index, future in enumerate(as_completed(futures), start=1):
            vbm_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # preserve every job result before stopping
                result = {"vbm_id": vbm_id, "status": "FAILED", "error": str(exc)}
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                failed = sum(x.get("status") != "OK" for x in results)
                log(f"{name}: {index}/{len(futures)}, failed={failed}")
    frame = pd.DataFrame(results).sort_values("vbm_id")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(status_path, sep="\t", index=False)
    return frame


def exclude_failed_triplets(
    rows: pd.DataFrame,
    status: pd.DataFrame,
    stage: str,
    tables: Path,
) -> pd.DataFrame:
    failed_ids = set(status.loc[status["status"].ne("OK"), "vbm_id"].astype(str))
    failed_triplets = set(rows.loc[rows["vbm_id"].isin(failed_ids), "triplet_id"].astype(str))
    excluded = rows[rows["triplet_id"].isin(failed_triplets)].copy()
    if not excluded.empty:
        excluded["exclusion_stage"] = stage
        excluded["exclusion_reason"] = excluded["vbm_id"].map(
            lambda value: "direct_stage_failure" if value in failed_ids else "matched_triplet_member_failure"
        )
        out = tables / "excluded_preprocessing_triplets.tsv"
        if out.exists():
            previous = pd.read_csv(out, sep="\t")
            excluded = pd.concat([previous, excluded], ignore_index=True).drop_duplicates(
                ["triplet_id", "vbm_id"], keep="last"
            )
        excluded.to_csv(out, sep="\t", index=False)
    kept = rows[~rows["triplet_id"].isin(failed_triplets)].copy()
    if failed_triplets:
        log(f"{stage}: removed {len(failed_triplets)} complete matched triplets")
    counts = kept.groupby("label").size().to_dict()
    if len(set(counts.values())) != 1 or set(counts) != {"CN", "MCI", "Dementia"}:
        raise RuntimeError(f"Unbalanced groups after {stage}: {counts}")
    return kept


def merge_template(
    inputs: list[Path],
    output_4d: Path,
    output_mean: Path,
    output_final: Path,
    log_path: Path,
    env: dict[str, str],
) -> None:
    if good(output_final):
        log(f"SKIP template merge: {output_final.name}")
        return
    run_cmd(["fslmerge", "-t", output_4d, *inputs], log_path, cwd=None, env=env)
    run_cmd(["fslmaths", output_4d, "-Tmean", output_mean], log_path, cwd=None, env=env)
    flipped = output_mean.with_name(output_mean.name.replace(".nii.gz", "_flipped.nii.gz"))
    run_cmd(["fslswapdim", output_mean, "-x", "y", "z", flipped], log_path, cwd=None, env=env)
    run_cmd(["fslmaths", output_mean, "-add", flipped, "-div", "2", output_final], log_path, cwd=None, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--fsldir", type=Path, default=DEFAULT_FSLDIR)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    work = args.work_dir.resolve()
    fslvbm = work / "fslvbm"
    struc = fslvbm / "struc"
    stats = fslvbm / "stats"
    tables = work / "tables"
    logs = work / "logs"
    status_dir = work / "status"
    provenance = work / "provenance"
    for path in (struc, stats, tables, logs, status_dir, provenance):
        path.mkdir(parents=True, exist_ok=True)
    env = configure_env(args.fsldir)

    rows = pd.read_csv(tables / "sample_plan.tsv", sep="\t")
    if len(rows) == 0 or rows["vbm_id"].duplicated().any() or rows["subject_id"].duplicated().any():
        raise RuntimeError("Invalid or duplicate sample plan")
    missing = [value for value in rows.vbm_id if not good(fslvbm / f"{value}.nii.gz")]
    if missing:
        raise RuntimeError(f"Missing reoriented NIfTI inputs: {len(missing)}")

    def bet_worker(row) -> dict[str, object]:
        base = f"{row.vbm_id}_struc"
        source = fslvbm / f"{row.vbm_id}.nii.gz"
        structural = struc / f"{base}.nii.gz"
        brain = struc / f"{base}_brain.nii.gz"
        mask = struc / f"{base}_brain_mask.nii.gz"
        job_log = logs / "bet" / f"{row.vbm_id}.log"
        try:
            if not good(structural):
                run_cmd(["imcp", source, struc / base], job_log, cwd=None, env=env)
            if not (good(brain) and good(mask)):
                run_cmd(["bet", base, f"{base}_brain", "-B", "-f", "0.5", "-m"], job_log, cwd=struc, env=env)
            values = image_stats(brain)
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "OK", "error": "", **values}
        except Exception as exc:
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "FAILED", "error": str(exc)}

    bet_status = run_parallel("BET-B", rows, bet_worker, args.workers, tables / "bet_status.tsv")
    rows = exclude_failed_triplets(rows, bet_status, "BET-B", tables)
    if bet_status["status"].eq("OK").sum() != len(rows):
        # Failed triplets may include successful partners; only retained rows matter.
        retained_status = bet_status[bet_status.vbm_id.isin(rows.vbm_id)]
        if not retained_status.status.eq("OK").all():
            raise RuntimeError("Retained BET set contains failures")

    def fast_worker(row) -> dict[str, object]:
        base = f"{row.vbm_id}_struc"
        brain_base = struc / f"{base}_brain"
        gm = struc / f"{base}_GM.nii.gz"
        pve0 = struc / f"{base}_brain_pve_0.nii.gz"
        pve1 = struc / f"{base}_brain_pve_1.nii.gz"
        pve2 = struc / f"{base}_brain_pve_2.nii.gz"
        job_log = logs / "fast" / f"{row.vbm_id}.log"
        try:
            if not (good(pve0) and good(pve1) and good(pve2)):
                run_cmd(["fast", "-R", "0.3", "-H", "0.1", brain_base], job_log, cwd=None, env=env)
            if not good(gm):
                run_cmd(["imcp", pve1, gm], job_log, cwd=None, env=env)
            images = [nib.load(str(path)) for path in (pve0, pve1, pve2)]
            voxel_volume = float(np.prod(images[0].header.get_zooms()[:3]))
            tissue_sums = [float(np.asarray(image.dataobj, dtype=np.float64).sum()) for image in images]
            return {
                "vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                "status": "OK", "error": "", "voxel_volume_mm3": voxel_volume,
                "csf_volume_mm3": tissue_sums[0] * voxel_volume,
                "gm_volume_mm3": tissue_sums[1] * voxel_volume,
                "wm_volume_mm3": tissue_sums[2] * voxel_volume,
                "tiv_mm3": sum(tissue_sums) * voxel_volume,
            }
        except Exception as exc:
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "FAILED", "error": str(exc)}

    fast_status = run_parallel("FAST", rows, fast_worker, args.workers, tables / "fast_tiv_status.tsv")
    rows = exclude_failed_triplets(rows, fast_status, "FAST", tables)
    fast_ok = fast_status[fast_status.status.eq("OK")]
    rows = rows.merge(
        fast_ok[["vbm_id", "csf_volume_mm3", "gm_volume_mm3", "wm_volume_mm3", "tiv_mm3"]],
        on="vbm_id", how="left", validate="one_to_one",
    )
    if rows.tiv_mm3.isna().any():
        raise RuntimeError("TIV missing after FAST")

    # This FSL installation distributes the tissue prior as an Analyze
    # .hdr/.img pair; FSL commands resolve it from the extension-free basename.
    standard_gm = args.fsldir / "data/standard/tissuepriors/avg152T1_gray"

    def affine_worker(row) -> dict[str, object]:
        base = f"{row.vbm_id}_struc"
        gm = struc / f"{base}_GM.nii.gz"
        output = struc / f"{base}_GM_to_T.nii.gz"
        job_log = logs / "template_affine" / f"{row.vbm_id}.log"
        try:
            if not good(output):
                run_cmd(["fsl_reg", gm, standard_gm, struc / f"{base}_GM_to_T", "-a"], job_log, cwd=None, env=env)
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "OK", "error": "", **image_stats(output)}
        except Exception as exc:
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "FAILED", "error": str(exc)}

    affine_status = run_parallel(
        "template affine registration", rows, affine_worker, args.workers,
        tables / "template_affine_status.tsv",
    )
    rows = exclude_failed_triplets(rows, affine_status, "template_affine", tables)

    affine_inputs = [struc / f"{value}_struc_GM_to_T.nii.gz" for value in rows.vbm_id]
    merge_template(
        affine_inputs,
        struc / "template_4D_GM_affine.nii.gz",
        struc / "template_GM_affine_mean.nii.gz",
        struc / "template_GM_init.nii.gz",
        logs / "template_merge_init.log", env,
    )

    def initial_nonlinear_worker(row) -> dict[str, object]:
        base = f"{row.vbm_id}_struc"
        gm = struc / f"{base}_GM.nii.gz"
        output = struc / f"{base}_GM_to_T_init.nii.gz"
        job_log = logs / "template_nonlinear" / f"{row.vbm_id}.log"
        try:
            if not good(output):
                run_cmd([
                    "fsl_reg", gm, struc / "template_GM_init.nii.gz",
                    struc / f"{base}_GM_to_T_init", "-fnirt",
                    "--config=GM_2_MNI152GM_2mm.cnf",
                ], job_log, cwd=None, env=env)
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "OK", "error": "", **image_stats(output)}
        except Exception as exc:
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "FAILED", "error": str(exc)}

    init_status = run_parallel(
        "template nonlinear registration", rows, initial_nonlinear_worker, args.workers,
        tables / "template_nonlinear_status.tsv",
    )
    rows_after_init = exclude_failed_triplets(rows, init_status, "template_nonlinear", tables)
    if len(rows_after_init) != len(rows):
        log("Rebuilding the final template without failed complete triplets")
    rows = rows_after_init
    nonlinear_inputs = [struc / f"{value}_struc_GM_to_T_init.nii.gz" for value in rows.vbm_id]
    merge_template(
        nonlinear_inputs,
        struc / "template_4D_GM.nii.gz",
        struc / "template_GM_mean.nii.gz",
        struc / "template_GM.nii.gz",
        logs / "template_merge_final.log", env,
    )

    def final_registration_worker(row) -> dict[str, object]:
        base = f"{row.vbm_id}_struc"
        gm = struc / f"{base}_GM.nii.gz"
        registered = struc / f"{base}_GM_to_template_GM.nii.gz"
        jacobian = struc / f"{base}_JAC_nl.nii.gz"
        modulated = struc / f"{base}_GM_to_template_GM_mod.nii.gz"
        job_log = logs / "final_registration" / f"{row.vbm_id}.log"
        try:
            if not (good(registered) and good(jacobian)):
                run_cmd([
                    "fsl_reg", gm, struc / "template_GM.nii.gz",
                    struc / f"{base}_GM_to_template_GM", "-fnirt",
                    f"--config=GM_2_MNI152GM_2mm.cnf --jout={jacobian}",
                ], job_log, cwd=None, env=env)
            if not good(modulated):
                run_cmd(["fslmaths", registered, "-mul", jacobian, modulated, "-odt", "float"], job_log, cwd=None, env=env)
            jac_stats = image_stats(jacobian)
            mod_stats = image_stats(modulated)
            return {
                "vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                "status": "OK", "error": "",
                **{f"jacobian_{key}": value for key, value in jac_stats.items()},
                **{f"modulated_gm_{key}": value for key, value in mod_stats.items()},
            }
        except Exception as exc:
            return {"vbm_id": row.vbm_id, "triplet_id": row.triplet_id, "label": row.label,
                    "status": "FAILED", "error": str(exc)}

    final_status = run_parallel(
        "final nonlinear registration/modulation", rows, final_registration_worker,
        args.workers, tables / "final_registration_status.tsv",
    )
    if not final_status.status.eq("OK").all():
        failed = final_status.loc[final_status.status.ne("OK"), "vbm_id"].tolist()
        raise RuntimeError(
            "Final registration failures require removal of complete triplets and a template rebuild: "
            + ", ".join(failed)
        )

    rows = rows.sort_values(["triplet_id", "analysis_role"]).reset_index(drop=True)
    rows.insert(0, "four_d_index_1based", np.arange(1, len(rows) + 1))
    rows.to_csv(tables / "analysis_order.tsv", sep="\t", index=False)
    rows.to_csv(tables / "sample_final_with_tiv.tsv", sep="\t", index=False)

    merge_log = logs / "merge_analysis_images.log"
    unmodulated = [struc / f"{value}_struc_GM_to_template_GM.nii.gz" for value in rows.vbm_id]
    modulated = [struc / f"{value}_struc_GM_to_template_GM_mod.nii.gz" for value in rows.vbm_id]
    if not good(stats / "GM_mod_merg.nii.gz"):
        run_cmd(["imcp", struc / "template_GM.nii.gz", stats / "template_GM.nii.gz"], merge_log, cwd=None, env=env)
        run_cmd(["fslmerge", "-t", stats / "GM_merg.nii.gz", *unmodulated], merge_log, cwd=None, env=env)
        run_cmd(["fslmerge", "-t", stats / "GM_mod_merg.nii.gz", *modulated], merge_log, cwd=None, env=env)
    if not good(stats / "GM_mask.nii.gz"):
        run_cmd(["fslmaths", stats / "GM_merg.nii.gz", "-Tmean", "-thr", "0.01", "-bin", stats / "GM_mask.nii.gz", "-odt", "char"], merge_log, cwd=None, env=env)
    if not good(stats / "GM_mod_merg_s2.nii.gz"):
        run_cmd(["fslmaths", stats / "GM_mod_merg.nii.gz", "-s", "2", stats / "GM_mod_merg_s2.nii.gz"], merge_log, cwd=None, env=env)
    if not good(stats / "GM_mod_merg_s3.nii.gz"):
        run_cmd(["fslmaths", stats / "GM_mod_merg.nii.gz", "-s", "3", stats / "GM_mod_merg_s3.nii.gz"], merge_log, cwd=None, env=env)
    for image in ("GM_merg", "GM_mod_merg", "GM_mod_merg_s2", "GM_mod_merg_s3"):
        output = stats / f"{image}_Tmean.nii.gz"
        if not good(output):
            run_cmd(["fslmaths", stats / f"{image}.nii.gz", "-Tmean", output], merge_log, cwd=None, env=env)

    counts = rows.groupby("label").size().to_dict()
    metadata = {
        "completed_at": stamp(),
        "workers": args.workers,
        "fsl_version": (args.fsldir / "etc/fslversion").read_text().strip()
        if (args.fsldir / "etc/fslversion").exists() else "unknown",
        "n_subjects": int(len(rows)),
        "n_triplets": int(rows.triplet_id.nunique()),
        "group_counts": counts,
        "brain_extraction": "BET -B -f 0.5 -m",
        "segmentation": "FAST -R 0.3 -H 0.1",
        "template": "balanced three-group symmetric study-specific FSL-VBM template",
        "analysis_images": [
            "GM_mod_merg (unsmoothed)", "GM_mod_merg_s2 (sigma=2 mm)",
            "GM_mod_merg_s3 (sigma=3 mm primary)",
        ],
        "tiv_definition": "sum of FAST CSF, GM and WM partial-volume estimates times native voxel volume",
    }
    (provenance / "vbm_preprocessing.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (status_dir / "vbm_preprocessing.ok").write_text(stamp() + "\n")
    log(f"VBM preprocessing complete: {counts}")


if __name__ == "__main__":
    main()
