#!/usr/bin/env python3
"""Lightweight subject-resampling stability analysis for the primary SEFT results.

The image matrix is loaded once and the same adjusted voxelwise GLM used by the
primary analysis is refitted analytically.  This avoids writing 80 resampled 4-D
VBM files.  Only AAL3, sigma=3, wavelet denoising, and alpha=0.1 are evaluated.
The two stronger contrasts use the primary PC fraction c=0.20; CN--MCI uses
c=0.10 to assess the stability of its reported regional findings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


REPRO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = REPRO_ROOT / "outputs/adni3"
TOOLS = REPRO_ROOT / "tools/seft_fsl"
SEFT_ENV = Path(sys.executable).resolve().parent.parent
AAL3 = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
AAL3_LABELS = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.txt"

sys.path.insert(0, str(TOOLS))
from seft_fsl_lib import run_seft_analysis  # noqa: E402


PRIMARY = ["CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI"]
CONTRASTS = np.asarray([[1, 0, -1], [0, 1, -1], [1, -1, 0]], dtype=float)
PC_LEVEL = {
    "CN_gt_Dementia": 0.20,
    "MCI_gt_Dementia": 0.20,
    "CN_gt_MCI": 0.10,
}
SEED = 20260717


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def good(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def configure_process() -> None:
    os.environ["PATH"] = f"{SEFT_ENV / 'bin'}:" + os.environ.get("PATH", "")
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[key] = "1"


def make_replicates(rows: pd.DataFrame, n_bootstrap: int, n_delete: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {
        label: np.flatnonzero(rows.label.to_numpy() == label)
        for label in ("CN", "MCI", "Dementia")
    }
    records: list[dict[str, object]] = []
    for replicate in range(1, n_bootstrap + 1):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()
        ])
        for order, index in enumerate(selected, start=1):
            records.append({
                "resampling": "bootstrap", "replicate": replicate,
                "resampled_order_1based": order, "source_index_0based": int(index),
                "image_id": rows.iloc[index].image_id, "label": rows.iloc[index].label,
            })
    for replicate in range(1, n_delete + 1):
        kept_blocks = []
        for indices in groups.values():
            n_remove = max(1, int(round(0.10 * len(indices))))
            removed = set(rng.choice(indices, size=n_remove, replace=False).tolist())
            kept_blocks.append(np.asarray([index for index in indices if index not in removed], dtype=int))
        selected = np.concatenate(kept_blocks)
        for order, index in enumerate(selected, start=1):
            records.append({
                "resampling": "delete10", "replicate": replicate,
                "resampled_order_1based": order, "source_index_0based": int(index),
                "image_id": rows.iloc[index].image_id, "label": rows.iloc[index].label,
            })
    return pd.DataFrame(records)


def fit_glm_zmaps(
    y: np.ndarray, design: np.ndarray, selected: np.ndarray, contrast_matrix: np.ndarray
) -> tuple[np.ndarray, int, int]:
    x = design[selected, :]
    rank = int(np.linalg.matrix_rank(x))
    df = int(x.shape[0] - rank)
    if df <= 0:
        raise RuntimeError("Non-positive residual degrees of freedom")
    xtx_inverse = np.linalg.pinv(x.T @ x, rcond=1e-12)
    y_selected = y[selected, :]
    xty = x.T @ y_selected
    beta = xtx_inverse @ xty
    # This is the OLS residual sum of squares without materializing fitted data.
    sum_y2 = np.einsum("ij,ij->j", y_selected, y_selected, dtype=np.float64)
    residual_ss = sum_y2 - np.einsum("ij,ij->j", beta, xty)
    residual_variance = np.maximum(residual_ss / df, np.finfo(float).tiny)
    outputs = []
    for contrast in contrast_matrix:
        estimate = contrast @ beta
        contrast_variance = float(contrast @ xtx_inverse @ contrast)
        tstat = estimate / np.sqrt(residual_variance * contrast_variance)
        probability = t_dist.cdf(np.abs(tstat), df=df)
        # ppf is imported lazily to keep the t-to-z expression explicit here.
        from scipy.stats import norm

        zstat = np.sign(tstat) * norm.ppf(np.clip(probability, 1e-15, 1 - 1e-15))
        outputs.append(np.nan_to_num(zstat, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32))
    return np.stack(outputs), df, rank


def save_masked_map(values: np.ndarray, mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = values
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, reference.affine, header), str(path))


def load_pc_frame(path: Path, pc_level: float) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    return frame[
        (frame.method.str.lower() == "seft") & np.isclose(frame.pc_level, pc_level)
    ].copy()


def contrast_output_name(contrast: str) -> str:
    pc_level = PC_LEVEL[contrast]
    return contrast if np.isclose(pc_level, 0.20) else f"{contrast}_c{int(round(100 * pc_level)):03d}"


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def plot_stability(summary: pd.DataFrame, output: Path) -> None:
    contrasts = PRIMARY
    display = {
        "CN_gt_Dementia": "CN > dementia",
        "MCI_gt_Dementia": "MCI > dementia",
        "CN_gt_MCI": "CN > MCI",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    for axis, contrast in zip(axes, contrasts):
        block = summary[summary.contrast.eq(contrast)].copy()
        block["max_frequency"] = block[["bootstrap_frequency", "delete10_frequency"]].max(axis=1)
        block = block[block.full_sample_selected.eq(1)]
        block = block.sort_values("max_frequency", ascending=False).head(35)
        if block.empty:
            axis.text(0.5, 0.5, "No regions selected", ha="center", va="center")
            axis.set_axis_off()
            continue
        matrix = block[["bootstrap_frequency", "delete10_frequency"]].to_numpy(float)
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
        axis.set_xticks([0, 1], ["Bootstrap", "Delete 10%"], rotation=25, ha="right")
        labels = block.region_label.tolist()
        axis.set_yticks(np.arange(len(block)), labels, fontsize=7)
        axis.set_title(f"{display[contrast]} ($c={PC_LEVEL[contrast]:.2f}$)")
        for row_index in range(matrix.shape[0]):
            for column_index in range(2):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}",
                          ha="center", va="center", fontsize=6,
                          color="white" if matrix[row_index, column_index] > 0.55 else "black")
    figure.colorbar(image, ax=axes, label="Selection frequency", shrink=0.65)
    figure.suptitle(
        "Selection frequencies of full-sample AAL3 SEFT discoveries "
        "(sigma=3, bw=5 voxels)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    global SEFT_ENV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--seft-env", type=Path, default=SEFT_ENV)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--n-delete", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    SEFT_ENV = args.seft_env.resolve()
    configure_process()
    work = args.work_dir.resolve()
    primary = work / "results/application_methods"
    if not good(primary / "analysis_complete.ok"):
        raise SystemExit("Primary analysis has not completed")
    output = work / "results/stability_primary"
    tables = output / "tables"
    logs = output / "logs"
    scratch = output / "scratch_zmaps"
    for directory in (tables, logs, scratch):
        directory.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(primary / "tables/analysis_subjects.tsv", sep="\t")
    design_frame = pd.read_csv(primary / "tables/design_matrix.tsv", sep="\t")
    design = design_frame.to_numpy(float)
    contrast_matrix = np.zeros((3, design.shape[1]), dtype=float)
    contrast_matrix[:, :3] = CONTRASTS
    definitions_path = tables / "resampling_definitions.tsv.gz"
    if definitions_path.exists():
        definitions = pd.read_csv(definitions_path, sep="\t")
    else:
        definitions = make_replicates(rows, args.n_bootstrap, args.n_delete, args.seed)
        definitions.to_csv(definitions_path, sep="\t", index=False, compression="gzip")

    mask_img = nib.load(str(work / "fslvbm/stats/GM_mask.nii.gz"))
    mask = np.asarray(mask_img.dataobj) > 0
    data_img = nib.load(str(work / "fslvbm/stats/GM_mod_merg_s3.nii.gz"))
    if data_img.shape[3] != len(rows):
        raise RuntimeError("4-D image order does not match analysis table")
    log(f"loading sigma=3 image matrix ({len(rows)} subjects, {int(mask.sum())} mask voxels)")
    full_data = data_img.get_fdata(dtype=np.float32)
    y = np.asarray(full_data[mask, :].T, dtype=np.float32, order="C")
    del full_data
    log(f"image matrix loaded: {y.nbytes / 2**20:.1f} MiB")

    # Validate the analytic refit against FSL randomise before resampling.
    validation_path = tables / "full_sample_glm_validation.tsv"
    if not validation_path.exists():
        full_z, full_df, full_rank = fit_glm_zmaps(y, design, np.arange(len(rows)), contrast_matrix)
        validation = []
        for index, contrast in enumerate(PRIMARY, start=1):
            fsl_path = primary / "randomise" / f"three_group_sigma3_tstat{index}.nii.gz"
            fsl_t = np.asarray(nib.load(str(fsl_path)).dataobj)[mask]
            probability = t_dist.cdf(np.abs(fsl_t), df=full_df)
            from scipy.stats import norm

            fsl_z = np.sign(fsl_t) * norm.ppf(np.clip(probability, 1e-15, 1 - 1e-15))
            difference = full_z[index - 1].astype(float) - fsl_z
            validation.append({
                "contrast": contrast, "n": len(rows), "rank": full_rank, "residual_df": full_df,
                "pearson_r_z": float(np.corrcoef(full_z[index - 1], fsl_z)[0, 1]),
                "mean_absolute_z_difference": float(np.mean(np.abs(difference))),
                "max_absolute_z_difference": float(np.max(np.abs(difference))),
            })
        pd.DataFrame(validation).to_csv(validation_path, sep="\t", index=False)
        if min(row["pearson_r_z"] for row in validation) < 0.99999:
            raise RuntimeError("Analytic GLM does not reproduce the primary FSL t maps")

    replicate_keys = (
        definitions[["resampling", "replicate"]].drop_duplicates()
        .sort_values(["resampling", "replicate"]).itertuples(index=False, name=None)
    )
    replicate_keys = list(replicate_keys)

    def process_replicate(resampling: str, replicate: int) -> list[pd.DataFrame]:
        replicate_dir = output / resampling / f"replicate_{replicate:03d}"
        done = replicate_dir / "complete.ok"
        frames = []
        expected = [
            replicate_dir / contrast_output_name(contrast) / "tables"
            / f"{resampling}_{replicate:03d}_{contrast_output_name(contrast)}_region_results.tsv"
            for contrast in PRIMARY
        ]
        if good(done) and all(good(path) for path in expected):
            for contrast, path in zip(PRIMARY, expected):
                frame = load_pc_frame(path, PC_LEVEL[contrast])
                frame.insert(0, "contrast", contrast)
                frames.append(frame)
            return frames
        selected = definitions[
            definitions.resampling.eq(resampling) & definitions.replicate.eq(replicate)
        ].sort_values("resampled_order_1based").source_index_0based.to_numpy(int)
        zmaps, df, rank = fit_glm_zmaps(y, design, selected, contrast_matrix)
        metadata = {"resampling": resampling, "replicate": replicate, "n": int(len(selected)),
                    "rank": rank, "residual_df": df, "fixed_mirror_seed": args.seed}
        replicate_dir.mkdir(parents=True, exist_ok=True)
        (replicate_dir / "glm_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for contrast_index, (contrast, result_path) in enumerate(zip(PRIMARY, expected)):
            if good(result_path):
                frame = load_pc_frame(result_path, PC_LEVEL[contrast])
                frame.insert(0, "contrast", contrast)
                frames.append(frame)
                continue
            output_name = contrast_output_name(contrast)
            prefix = f"{resampling}_{replicate:03d}_{output_name}"
            input_map = scratch / f"{prefix}.nii.gz"
            save_masked_map(zmaps[contrast_index], mask, mask_img, input_map)
            run_dir = replicate_dir / output_name
            result = run_seft_analysis(
                zmap=input_map, tstat=None, design_mat=None,
                atlas=AAL3, atlas_labels=AAL3_LABELS,
                mask=work / "fslvbm/stats/GM_mask.nii.gz",
                out_dir=run_dir, prefix=prefix, alpha=0.1,
                pc_levels=[PC_LEVEL[contrast]],
                simes=False, seed=args.seed, denoise="wavelet", bandwidth=5.0,
                lambda_value=0.5, neighbor_range=10, score_clip_c=0.99,
                save_internals=False, rscript=str(SEFT_ENV / "bin/Rscript"), build_maps=False,
            )
            frame = load_pc_frame(result.region_results, PC_LEVEL[contrast])
            frame.insert(0, "contrast", contrast)
            frames.append(frame)
            input_map.unlink(missing_ok=True)
            Path(result.signed_z).unlink(missing_ok=True)
        done.write_text(stamp() + "\n")
        return frames

    all_frames: list[pd.DataFrame] = []
    log(f"running/resuming {len(replicate_keys)} resamples with {args.jobs} concurrent replicate jobs")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(process_replicate, resampling, int(replicate)): (resampling, int(replicate))
            for resampling, replicate in replicate_keys
        }
        for future in as_completed(futures):
            resampling, replicate = futures[future]
            frames = future.result()
            for frame in frames:
                frame.insert(0, "replicate", replicate)
                frame.insert(0, "resampling", resampling)
                all_frames.append(frame)
            log(f"complete: {resampling} {replicate:03d}")

    region_long = pd.concat(all_frames, ignore_index=True)
    region_long.to_csv(tables / "stability_region_results_long.tsv.gz", sep="\t", index=False, compression="gzip")
    full = pd.read_csv(primary / "tables/seft_simes_region_results_long.tsv", sep="\t")
    full = pd.concat([
        full[
            full.atlas.eq("aal3") & full.variant.eq("sigma3")
            & full.contrast.eq(contrast)
            & np.isclose(full.pc_level, PC_LEVEL[contrast])
            & full.method.str.lower().eq("seft")
            & np.isclose(full.alpha, 0.1)
        ]
        for contrast in PRIMARY
    ], ignore_index=True)
    full_lookup = {
        (row.contrast, int(row.region_id)): int(row.significant_recomputed)
        for row in full.itertuples()
    }
    aggregate = region_long.groupby(
        ["resampling", "contrast", "region_id", "region_label"], as_index=False
    ).agg(
        n_replicates=("replicate", "nunique"), selection_frequency=("significant", "mean"),
        e_value_median=("e_value", "median"), e_value_q10=("e_value", lambda x: x.quantile(0.10)),
        e_value_q90=("e_value", lambda x: x.quantile(0.90)),
        pc_p_value_median=("pc_p_value", "median"),
    )
    wide = aggregate.pivot(index=["contrast", "region_id", "region_label"], columns="resampling")
    wide.columns = [f"{resampling}_{metric}" for metric, resampling in wide.columns]
    wide = wide.reset_index()
    for resampling in ("bootstrap", "delete10"):
        frequency = f"{resampling}_selection_frequency"
        if frequency not in wide:
            wide[frequency] = np.nan
    wide["bootstrap_frequency"] = wide["bootstrap_selection_frequency"]
    wide["delete10_frequency"] = wide["delete10_selection_frequency"]
    wide["full_sample_selected"] = [
        full_lookup.get((contrast, int(region_id)), 0)
        for contrast, region_id in zip(wide.contrast, wide.region_id)
    ]
    def stability_class(row: pd.Series) -> str:
        frequencies = [row.bootstrap_frequency, row.delete10_frequency]
        if min(frequencies) >= 0.70:
            return "stable"
        if min(frequencies) < 0.50:
            return "unstable_exploratory"
        return "moderately_stable"
    wide["stability_class"] = wide.apply(stability_class, axis=1)
    wide.to_csv(tables / "region_stability_summary.tsv", sep="\t", index=False)

    replicate_summary = []
    for (resampling, replicate, contrast), block in region_long.groupby(["resampling", "replicate", "contrast"]):
        selected = set(block.loc[block.significant.eq(1), "region_id"].astype(int))
        full_selected = {region for (name, region), value in full_lookup.items() if name == contrast and value == 1}
        replicate_summary.append({
            "resampling": resampling, "replicate": replicate, "contrast": contrast,
            "discovered_regions": len(selected), "full_sample_discovered_regions": len(full_selected),
            "jaccard_with_full_sample": jaccard(selected, full_selected),
        })
    replicate_summary_frame = pd.DataFrame(replicate_summary)
    replicate_summary_frame.to_csv(tables / "replicate_summary.tsv", sep="\t", index=False)
    replicate_summary_frame.groupby(["resampling", "contrast"], as_index=False).agg(
        n_replicates=("replicate", "nunique"), min_discoveries=("discovered_regions", "min"),
        median_discoveries=("discovered_regions", "median"), max_discoveries=("discovered_regions", "max"),
        median_jaccard=("jaccard_with_full_sample", "median"),
        min_jaccard=("jaccard_with_full_sample", "min"), max_jaccard=("jaccard_with_full_sample", "max"),
    ).to_csv(tables / "stability_overall_summary.tsv", sep="\t", index=False)
    figure_path = work / "manuscript/figures/adni_resampling_stability_summary.png"
    plot_stability(wide, figure_path)
    shutil.rmtree(scratch, ignore_errors=True)
    metadata = {
        "completed_at": stamp(), "seed": args.seed, "fixed_mirror_seed": args.seed,
        "n_bootstrap": args.n_bootstrap, "n_delete10": args.n_delete,
        "settings": {"atlas": "AAL3", "sigma_mm": 3, "wavelet": True,
                     "pc_fraction_c_by_contrast": PC_LEVEL, "alpha": 0.1,
                     "bandwidth_voxels": 5.0,
                     "neighbor_range_voxels": 10},
        "interpretation": "Selection stability only; not null calibration or an error-rate estimate.",
    }
    (work / "provenance/stability_analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "analysis_complete.ok").write_text(stamp() + "\n")
    log("Stability analysis complete")


if __name__ == "__main__":
    main()
