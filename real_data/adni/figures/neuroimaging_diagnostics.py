#!/usr/bin/env python3
"""Generate real-map model checks and paper-facing application figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis, median_abs_deviation, norm, skew, spearmanr
from nilearn import datasets as nilearn_datasets
from nilearn import plotting as nilearn_plotting
from nilearn import surface as nilearn_surface


REPRO_ROOT = Path(__file__).resolve().parents[3]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from real_data.adni.metadata_files import resolve_metadata_file


DEFAULT_WORK = REPRO_ROOT / "outputs/adni3"
DEFAULT_DATA = REPRO_ROOT / "real_data/adni/input"
DXSUM: Path | None = None
AAL3 = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
AAL3_LABELS = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.txt"
PRIMARY = ["CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI"]
DISPLAY_NAMES = {
    "CN_gt_Dementia": "CN > dementia",
    "MCI_gt_Dementia": "MCI > dementia",
    "CN_gt_MCI": "CN > MCI",
}
SEED = 20260717
DISPLAY_Z_THRESHOLD = 2.0
DISPLAY_Z_MAX = 8.0
MNI_T1 = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "data/standard/MNI152_T1_2mm_brain.nii.gz"


def good(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def load(path: Path, dtype=np.float32) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def axial_mni_z(reference: Path, z_index: int) -> float:
    """Return the MNI z coordinate at the centre of an axial voxel plane."""
    image = nib.load(str(reference))
    centre = np.asarray([(size - 1) / 2 for size in image.shape[:3]] + [1.0])
    centre[2] = z_index
    return float((image.affine @ centre)[2])


def label_lookup() -> dict[int, str]:
    output: dict[int, str] = {}
    for line in AAL3_LABELS.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            output[int(float(fields[0]))] = fields[1]
    return output


def finite_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return data[mask & np.isfinite(data)].astype(float, copy=False)


def summary_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n_voxels": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(np.median(values)),
        "mad_normal_consistent": float(median_abs_deviation(values, scale="normal")),
        "skewness": float(skew(values, bias=False)),
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "fraction_abs_z_ge_1p96": float(np.mean(np.abs(values) >= 1.96)),
        "fraction_abs_z_ge_2p576": float(np.mean(np.abs(values) >= 2.576)),
        "fraction_abs_z_ge_3": float(np.mean(np.abs(values) >= 3)),
        "fraction_abs_z_ge_4": float(np.mean(np.abs(values) >= 4)),
        "fraction_abs_z_ge_5": float(np.mean(np.abs(values) >= 5)),
        "q001": float(np.quantile(values, 0.001)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def sampled(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= n:
        return values.copy()
    return rng.choice(values, n, replace=False)


def save_publication_figure(fig: plt.Figure, figures: Path, stem: str, *, tight: bool = True) -> None:
    kwargs = {"facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(figures / f"{stem}.pdf", dpi=400, **kwargs)


def z_model_check(
    out: Path, zmaps: dict[str, Path], mask: np.ndarray, atlas: np.ndarray,
    labels: dict[int, str], tables: Path, figures: Path,
) -> None:
    rng = np.random.default_rng(SEED)
    whole_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 10.8))
    for row_index, contrast in enumerate(PRIMARY):
        raw_data = load(zmaps[contrast])
        raw_values = finite_values(raw_data, mask)
        whole_rows.append({
            "contrast": contrast, "map_stage": "displayed_signed_z", **summary_stats(raw_values)
        })
        for region_id in sorted(int(value) for value in np.unique(atlas[mask]) if value > 0):
            region_values = finite_values(raw_data, mask & atlas.__eq__(region_id))
            if region_values.size:
                region_rows.append({
                    "contrast": contrast, "region_id": region_id,
                    "region_label": labels.get(region_id, f"region_{region_id}"),
                    **summary_stats(region_values),
                })

        prefix = f"sigma3_{contrast}_aal3"
        model_path = out / "seft_runs" / prefix / "maps/claw_internal" / f"{prefix}_z_model.nii.gz"
        model_data = load(model_path, dtype=np.float64)
        model_active = mask & (atlas > 0) & np.isfinite(model_data) & (model_data != 0)
        values = model_data[model_active]
        whole_rows.append({
            "contrast": contrast, "map_stage": "claw_model_input", **summary_stats(values)
        })
        sample = np.sort(sampled(values, min(120_000, values.size), rng))
        theoretical = norm.ppf((np.arange(sample.size) + 0.5) / sample.size)
        ax = axes[row_index, 0]
        ax.plot(theoretical, sample, color="#2B6CB0", linewidth=1.2)
        low, high = float(theoretical[0]), float(theoretical[-1])
        ax.plot([low, high], [low, high], color="0.20", linestyle="--", linewidth=1.1, label="N(0,1)")
        ax.axhline(0, color="0.8", linewidth=0.7)
        ax.axvline(0, color="0.8", linewidth=0.7)
        ax.set(
            xlabel="Reference normal quantile",
            ylabel="Observed model-input Z quantile",
            title=f"{DISPLAY_NAMES[contrast]}: QQ plot",
        )
        ax.title.set_fontsize(12.5)
        ax.title.set_fontweight("semibold")
        ax.xaxis.label.set_fontsize(11.5)
        ax.yaxis.label.set_fontsize(11.5)
        ax.tick_params(labelsize=10.5)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        ax.text(
            0.98, 0.04,
            rf"mean = {np.mean(values):.2f}" + "\n" +
            rf"median = {np.median(values):.2f}" + "\n" +
            rf"SD = {np.std(values, ddof=1):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.9},
        )

        ax = axes[row_index, 1]
        grid_low, grid_high = np.quantile(sample, [0.001, 0.999])
        grid_low, grid_high = min(grid_low, -5), max(grid_high, 5)
        grid = np.linspace(grid_low, grid_high, 500)
        density_sample = sampled(values, min(50_000, values.size), rng)
        kde = gaussian_kde(density_sample)
        ax.hist(density_sample, bins=120, density=True, color="#9ECAE1", alpha=0.48,
                label="Observed histogram")
        ax.plot(grid, kde(grid), color="#2171B5", linewidth=1.8, label="Observed KDE")
        ax.plot(grid, norm.pdf(grid), color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
        ax.set(
            xlabel="Model-input Z", ylabel="Density",
            title=f"{DISPLAY_NAMES[contrast]}: distribution",
        )
        ax.title.set_fontsize(12.5)
        ax.title.set_fontweight("semibold")
        ax.xaxis.label.set_fontsize(11.5)
        ax.yaxis.label.set_fontsize(11.5)
        ax.tick_params(labelsize=10.5)
        ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout(pad=1.0, h_pad=1.4, w_pad=1.2)
    save_publication_figure(fig, figures, "zmap_model_check_qq_hist")
    plt.close(fig)
    pd.DataFrame(whole_rows).to_csv(tables / "zmap_distribution_summary.tsv", sep="\t", index=False)
    pd.DataFrame(region_rows).to_csv(tables / "zmap_distribution_by_aal3_region.tsv", sep="\t", index=False)


def ecdf_xy(values: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    if values.size > max_points:
        indices = np.linspace(0, values.size - 1, max_points).astype(int)
        values = values[indices]
        ranks = (indices + 1) / (indices[-1] + 1)
    else:
        ranks = np.arange(1, values.size + 1) / values.size
    return values, ranks


def robust_unit_scale(values: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Map the 1st--99th percentile interval to [0, 1] without clipping tails."""
    lower, upper = np.quantile(values[np.isfinite(values)], [low, high])
    if upper <= lower:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / (upper - lower)


def fsl_slice_panel(
    metric: np.ndarray,
    reference: Path,
    slice_specs: tuple[tuple[str, int], ...],
    scratch: Path,
    stem: str,
    positive_range: tuple[float, float],
    negative_range: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Render axial overlays on the FSL MNI152 grayscale anatomy."""
    reference_image = nib.load(str(reference))
    metric_path = scratch / f"{stem}_unit.nii.gz"
    rendered_path = scratch / f"{stem}_rendered.nii.gz"
    nib.save(
        nib.Nifti1Image(metric.astype(np.float32), reference_image.affine, reference_image.header),
        str(metric_path),
    )
    overlay_command = [
        "overlay", "1", "0", str(MNI_T1), "3000", "8000",
        str(metric_path), f"{positive_range[0]:g}", f"{positive_range[1]:g}",
    ]
    if negative_range is not None:
        overlay_command.extend([
            str(metric_path), f"{negative_range[0]:g}", f"{negative_range[1]:g}"
        ])
    overlay_command.append(str(rendered_path))
    subprocess.run(
        overlay_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    inverse_affine = np.linalg.inv(reference_image.affine)
    plane_index = {"x": 0, "y": 1, "z": 2}
    output: list[np.ndarray] = []
    for plane, coordinate in slice_specs:
        world = np.asarray([0.0, 0.0, 0.0, 1.0])
        world[plane_index[plane]] = coordinate
        voxel = int(round((inverse_affine @ world)[plane_index[plane]]))
        png_path = scratch / f"{stem}_{plane}{coordinate:+d}.png"
        subprocess.run(
            [
                "slicer", str(rendered_path), "-u",
                f"-{plane}", f"-{voxel}", str(png_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output.append(plt.imread(png_path))
    return output


def crop_fsl_sagittal(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Crop black FSL margins so sagittal panels match the other views."""
    visible = np.any(image[..., :3] > 0.01, axis=2)
    rows, columns = np.where(visible)
    if rows.size == 0:
        return image
    row_start = max(0, int(rows.min()) - padding)
    row_stop = min(image.shape[0], int(rows.max()) + padding + 1)
    column_start = max(0, int(columns.min()) - padding)
    column_stop = min(image.shape[1], int(columns.max()) + padding + 1)
    return image[row_start:row_stop, column_start:column_stop]


def claw_diagnostics(
    out: Path, mask: np.ndarray, atlas: np.ndarray, labels: dict[int, str],
    main_regions: pd.DataFrame, tables: Path, figures: Path,
) -> None:
    diagnostic_rows: list[dict[str, object]] = []
    regional_rows: list[dict[str, object]] = []
    binned_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    slice_specs = (
        ("x", -35), ("y", -30), ("z", 0),
        ("x", 35), ("y", 10), ("z", 30),
    )
    panel_titles = (
        r"Local $\pi$",
        r"$R$ evidence",
        r"Observed $Z$ map",
    )

    fig = plt.figure(figsize=(15.2, 8.3))
    grid = fig.add_gridspec(
        3, 4, width_ratios=(1, 1, 1, 1),
        left=0.070, right=0.985, bottom=0.085, top=0.970,
        wspace=0.14, hspace=0.25,
    )
    axes = np.empty(3, dtype=object)
    spatial_axes: list[list[list[plt.Axes]]] = []
    for row_index in range(3):
        axes[row_index] = fig.add_subplot(grid[row_index, 0])
        spatial_row: list[list[plt.Axes]] = []
        for panel_index, column_index in enumerate((1, 2, 3)):
            title_axis = fig.add_subplot(grid[row_index, column_index])
            title_axis.set_axis_off()
            if row_index == 0:
                title_axis.set_title(
                    panel_titles[panel_index], fontsize=15, fontweight="bold", pad=5
                )
            mini_grid = grid[row_index, column_index].subgridspec(
                2, 3, width_ratios=(1.4, 1, 1),
                wspace=0.012, hspace=0.012,
            )
            spatial_row.append([
                fig.add_subplot(mini_grid[slice_index // 3, slice_index % 3])
                for slice_index in range(6)
            ])
        spatial_axes.append(spatial_row)

    pooled_evidence: list[np.ndarray] = []
    for contrast in PRIMARY:
        prefix = f"sigma3_{contrast}_aal3"
        r_path = out / "seft_runs" / prefix / "maps/claw_internal" / f"{prefix}_R.nii.gz"
        r_data = load(r_path, dtype=np.float64)
        values = -np.log10(np.clip(r_data[mask & (atlas > 0)], np.finfo(float).tiny, None))
        pooled_evidence.append(values[np.isfinite(values) & (values > 0)])
    evidence_reference = float(np.quantile(np.concatenate(pooled_evidence), 0.99))

    with tempfile.TemporaryDirectory(prefix="claw_score_fsl_") as scratch_name:
        scratch = Path(scratch_name)
        for row_index, contrast in enumerate(PRIMARY):
            prefix = f"sigma3_{contrast}_aal3"
            internal = out / "seft_runs" / prefix / "maps/claw_internal"
            names = ("z_model", "z_til", "pi", "log_f", "log_f_til", "R", "R_til")
            paths = {name: internal / f"{prefix}_{name}.nii.gz" for name in names}
            missing = [path for path in paths.values() if not good(path)]
            if missing:
                raise RuntimeError(f"Missing SEFT-CO internal map: {missing[0]}")
            data = {name: load(path, dtype=np.float64) for name, path in paths.items()}
            active = mask & (atlas > 0)
            z = data["z_model"][active]
            z_til = data["z_til"][active]
            pi = data["pi"][active]
            log_f = data["log_f"][active]
            log_f_til = data["log_f_til"][active]
            r_obs = data["R"][active]
            r_til = data["R_til"][active]
            observed_clip = np.log1p(-pi) + norm.logpdf(z) - log_f >= np.log(0.99)
            mirror_clip = np.log1p(-pi) + norm.logpdf(z_til) - log_f_til >= np.log(0.99)

            for score_type, score, clip in (
                ("observed_R", r_obs, observed_clip),
                ("mirror_R_til", r_til, mirror_clip),
            ):
                finite = np.isfinite(score)
                finite_score = score[finite]
                diagnostic_rows.append({
                    "contrast": contrast, "score_type": score_type,
                    "n_voxels": int(score.size),
                    "nonfinite_fraction": float(np.mean(~finite)),
                    "zero_fraction": float(np.mean(finite & (score == 0))),
                    "negative_fraction": float(np.mean(finite & (score < 0))),
                    "extreme_ge_1e6_fraction": float(np.mean(finite & (score >= 1e6))),
                    "score_clipping_fraction": float(np.mean(clip)),
                    "median": float(np.median(finite_score)),
                    "q90": float(np.quantile(finite_score, 0.9)),
                    "q99": float(np.quantile(finite_score, 0.99)),
                    "q999": float(np.quantile(finite_score, 0.999)),
                    "max": float(np.max(finite_score)),
                })
            diagnostic_rows.append({
                "contrast": contrast, "score_type": "pi", "n_voxels": int(pi.size),
                "nonfinite_fraction": float(np.mean(~np.isfinite(pi))),
                "zero_fraction": float(np.mean(pi == 0)),
                "negative_fraction": float(np.mean(pi < 0)),
                "extreme_ge_1e6_fraction": 0.0,
                "score_clipping_fraction": float(np.mean(np.isclose(pi, 0.499, atol=1e-7))),
                "median": float(np.median(pi)), "q90": float(np.quantile(pi, 0.9)),
                "q99": float(np.quantile(pi, 0.99)),
                "q999": float(np.quantile(pi, 0.999)), "max": float(np.max(pi)),
            })

            main = main_regions[
                main_regions.contrast.eq(contrast) & main_regions.method.eq("seft")
            ]
            main_lookup = main.set_index("region_id")[
                ["e_value", "significant_recomputed"]
            ].to_dict("index")
            region_ids = sorted(int(value) for value in np.unique(atlas[active]) if value > 0)
            for region_id in region_ids:
                region_mask = active & atlas.__eq__(region_id)
                score = data["R"][region_mask]
                score = score[np.isfinite(score)]
                record = main_lookup.get(
                    region_id, {"e_value": np.nan, "significant_recomputed": 0}
                )
                regional_rows.append({
                    "contrast": contrast, "region_id": region_id,
                    "region_label": labels.get(region_id, f"region_{region_id}"),
                    "n_voxels": int(region_mask.sum()),
                    "median_R": float(np.median(score)),
                    "q90_R": float(np.quantile(score, 0.9)),
                    "q99_R": float(np.quantile(score, 0.99)),
                    "pc_e_value_c0p2": record["e_value"],
                    "seft_selected_alpha0p1": int(record["significant_recomputed"]),
                })

            score_valid = np.isfinite(r_obs) & (r_obs > 0) & np.isfinite(z)
            abs_z = np.abs(z[score_valid])
            x_all, y_all = abs_z, np.log(r_obs[score_valid])
            rho, rho_p = spearmanr(x_all, y_all)
            edges = np.quantile(x_all, np.linspace(0, 1, 21))
            bin_centers, medians = [], []
            lower_quartiles, upper_quartiles = [], []
            for bin_index in range(20):
                upper_test = (
                    x_all <= edges[bin_index + 1]
                    if bin_index == 19 else x_all < edges[bin_index + 1]
                )
                use = (x_all >= edges[bin_index]) & upper_test
                if use.any():
                    center = float(np.median(x_all[use]))
                    median = float(np.median(y_all[use]))
                    bin_centers.append(center)
                    medians.append(median)
                    lower_quartiles.append(float(np.quantile(y_all[use], 0.025)))
                    upper_quartiles.append(float(np.quantile(y_all[use], 0.975)))
                    binned_rows.append({
                        "contrast": contrast, "bin": bin_index + 1,
                        "median_abs_z": center, "median_log_R": median,
                        "n_voxels": int(use.sum()),
                    })
            relation_rows.append({
                "contrast": contrast, "n_voxels": int(x_all.size),
                "spearman_rho_abs_z_log_R": float(rho),
                "spearman_p_value": float(rho_p),
                "lowest_abs_z_bin_median_log_R": float(medians[0]),
                "highest_abs_z_bin_median_log_R": float(medians[-1]),
                "expected_direction": "negative because smaller SEFT-CO R is more signal-like",
            })
            axes[row_index].fill_between(
                bin_centers, lower_quartiles, upper_quartiles,
                color="#9ECAE1", alpha=0.55, label="95% band",
            )
            axes[row_index].plot(
                bin_centers, medians, color="#E6550D", marker="o", markersize=3.5,
                linewidth=1.5, label="Median",
            )
            axes[row_index].set(
                xlabel="|Z|", ylabel="log(R)",
            )
            if row_index == 0:
                axes[row_index].set_title(
                    "Score--signal relation", fontsize=15, fontweight="semibold"
                )
            axes[row_index].xaxis.label.set_fontsize(14)
            axes[row_index].yaxis.label.set_fontsize(14)
            axes[row_index].tick_params(labelsize=12)
            axes[row_index].text(
                0.05, 0.08, rf"$\rho_s={rho:.2f}$",
                transform=axes[row_index].transAxes, fontsize=13, fontweight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            if row_index == 0:
                axes[row_index].legend(frameon=False, fontsize=11)

            raw_z = load(
                out / "zmaps" / f"sigma3_{contrast}_signed_z.nii.gz",
                dtype=np.float64,
            )
            metric_maps = (
                np.where(active, np.clip(data["pi"], 0, 0.499), 0),
                np.where(
                    active,
                    np.clip(
                        -np.log10(np.clip(data["R"], np.finfo(float).tiny, None)),
                        0, evidence_reference,
                    ),
                    0,
                ),
                np.where(active, np.clip(raw_z, -DISPLAY_Z_MAX, DISPLAY_Z_MAX), 0),
            )
            display_ranges = (
                ((0.001, 0.499), None),
                ((0.001, evidence_reference), None),
                (
                    (DISPLAY_Z_THRESHOLD, DISPLAY_Z_MAX),
                    (-DISPLAY_Z_THRESHOLD, -DISPLAY_Z_MAX),
                ),
            )
            for panel_index, (metric_map, title, ranges) in enumerate(
                zip(metric_maps, panel_titles, display_ranges)
            ):
                images = fsl_slice_panel(
                    metric_map, paths["R"], slice_specs, scratch,
                    f"{row_index}_{panel_index}_{contrast}",
                    positive_range=ranges[0], negative_range=ranges[1],
                )
                for slice_index, (axis, image) in enumerate(
                    zip(spatial_axes[row_index][panel_index], images)
                ):
                    if slice_specs[slice_index][0] == "x":
                        image = crop_fsl_sagittal(image)
                    axis.imshow(image)
                    axis.axis("off")
                    axis.text(
                        0.04, 0.08,
                        f"{slice_specs[slice_index][0]}={slice_specs[slice_index][1]:+d}",
                        transform=axis.transAxes, color="white", fontsize=10.5,
                        bbox={
                            "facecolor": "black", "edgecolor": "none",
                            "alpha": 0.62, "pad": 1.0,
                        },
                    )

            axes[row_index].text(
                -0.28, 0.5, DISPLAY_NAMES[contrast],
                transform=axes[row_index].transAxes, rotation=90,
                ha="center", va="center", fontsize=14, fontweight="bold",
            )
    fsl_signed_cmap = LinearSegmentedColormap.from_list(
        "fsl_signed",
        [
            (0.00, "#c8f4ff"), (0.375, "#0000ff"), (0.499, "#ffffff"),
            (0.501, "#ffffff"), (0.625, "#ff0000"), (1.00, "#ffff00"),
        ],
    )
    colorbar_specs = (
        (1, Normalize(0, 0.499), "autumn", r"$\pi$", np.arange(0, 0.5, 0.1)),
        (
            2, Normalize(0, evidence_reference), "autumn",
            r"$-\log_{10}(R)$", np.arange(0, evidence_reference, 2),
        ),
        (
            3, Normalize(-DISPLAY_Z_MAX, DISPLAY_Z_MAX), fsl_signed_cmap,
            r"$Z$", np.arange(-DISPLAY_Z_MAX, DISPLAY_Z_MAX + 1, 4),
        ),
    )
    for column_index, normalizer, colormap, label, ticks in colorbar_specs:
        position = grid[2, column_index].get_position(fig)
        colorbar_width = position.width * 0.72
        colorbar_axis = fig.add_axes([
            position.x0 + (position.width - colorbar_width) / 2,
            position.y0 - 0.047,
            colorbar_width,
            0.012,
        ])
        colorbar = fig.colorbar(
            ScalarMappable(norm=normalizer, cmap=colormap),
            cax=colorbar_axis, orientation="horizontal", ticks=ticks,
        )
        colorbar.set_label(
            label, fontsize=14, fontweight="semibold", labelpad=3
        )
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.tick_params(labelsize=10.5, length=2.5, pad=2)
    save_publication_figure(fig, figures, "claw_score_model_check")
    save_publication_figure(fig, figures, "co_score_model_check_spatial")
    plt.close(fig)
    pd.DataFrame(diagnostic_rows).to_csv(
        tables / "claw_score_diagnostics.tsv", sep="\t", index=False
    )
    pd.DataFrame(regional_rows).to_csv(
        tables / "claw_score_by_aal3_region.tsv", sep="\t", index=False
    )
    pd.DataFrame(binned_rows).to_csv(
        tables / "claw_score_z_binned.tsv", sep="\t", index=False
    )
    pd.DataFrame(relation_rows).to_csv(
        tables / "claw_score_z_relation.tsv", sep="\t", index=False
    )


def score_cap_sensitivity_figure(tables: Path, figures: Path) -> None:
    path = tables / "score_cap_sensitivity_summary.tsv"
    if not good(path):
        raise RuntimeError(f"Missing score-cap sensitivity table: {path}")
    data = pd.read_csv(path, sep="\t")
    colors = {0.10: "#1B9E77", 0.20: "#D95F02", 0.30: "#7570B3"}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharex=True)
    cap_values = sorted(data.score_cap.unique())
    x_positions = np.arange(len(cap_values))
    for column, contrast in enumerate(PRIMARY):
        block = data[data.contrast.eq(contrast)].sort_values("score_cap")
        for pc_level, color in colors.items():
            line = block[np.isclose(block.pc_level, pc_level)]
            axes[0, column].plot(
                x_positions, line.discovered_regions, marker="o", linewidth=1.6,
                color=color, label=f"PC fraction c={pc_level:.2f}",
            )
            axes[1, column].plot(
                x_positions, line.jaccard_vs_cap0p99, marker="o", linewidth=1.6,
                color=color,
            )
        default_position = cap_values.index(0.99)
        axes[0, column].axvline(default_position, color="0.25", linestyle="--", linewidth=1)
        axes[1, column].axvline(default_position, color="0.25", linestyle="--", linewidth=1)
        axes[0, column].set_title(DISPLAY_NAMES[contrast])
        axes[0, column].set_ylabel("Discovered regions" if column == 0 else "")
        axes[1, column].set_ylabel("Jaccard vs cap=0.99" if column == 0 else "")
        axes[1, column].set_xlabel("SEFT-CO score cap")
        axes[1, column].set_xticks(x_positions, [f"{value:g}" for value in cap_values])
        axes[1, column].set_ylim(-0.03, 1.03)
        axes[1, column].grid(axis="y", color="0.9", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Sensitivity of regional SEFT findings to the SEFT-CO score cap", fontsize=15, y=1.035)
    fig.text(
        0.99, 0.008,
        "For CN > MCI at c=0.20/0.30, Jaccard=1 denotes two empty discovery sets.",
        ha="right", va="bottom", fontsize=8, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.94])
    save_publication_figure(fig, figures, "score_cap_sensitivity")
    plt.close(fig)


def signed_mip(data: np.ndarray, axis: int) -> np.ndarray:
    index = np.argmax(np.abs(data), axis=axis)
    return np.take_along_axis(data, np.expand_dims(index, axis=axis), axis=axis).squeeze(axis)


def zmap_figures(zmaps: dict[str, Path], mask: np.ndarray, figures: Path) -> None:
    if not good(MNI_T1):
        raise RuntimeError(f"Missing MNI structural background: {MNI_T1}")
    background = load(MNI_T1)
    reference_image = nib.load(str(next(iter(zmaps.values()))))
    background_image = nib.load(str(MNI_T1))
    if reference_image.shape[:3] != background_image.shape[:3] or not np.allclose(
        reference_image.affine, background_image.affine, atol=1e-5
    ):
        raise RuntimeError("MNI152 background geometry does not match the Z maps")
    bg_values = background[background > 0]
    bg_low, bg_high = np.quantile(bg_values, [0.02, 0.99])
    for contrast in PRIMARY:
        data = load(zmaps[contrast])
        slices = np.linspace(np.nonzero(mask)[2].min() + 5, np.nonzero(mask)[2].max() - 5, 12).astype(int)
        fig, axes = plt.subplots(3, 4, figsize=(12.4, 9.2))
        for axis, z_index in zip(axes.flat, slices):
            background_plane = np.rot90(background[:, :, z_index])
            overlay = np.rot90(np.where(
                mask[:, :, z_index] & (np.abs(data[:, :, z_index]) >= DISPLAY_Z_THRESHOLD),
                data[:, :, z_index], np.nan,
            ))
            axis.imshow(background_plane, cmap="gray", vmin=bg_low, vmax=bg_high)
            axis.imshow(
                overlay, cmap="RdBu_r", vmin=-DISPLAY_Z_MAX, vmax=DISPLAY_Z_MAX,
                interpolation="nearest", alpha=0.88,
            )
            mni_z = axial_mni_z(zmaps[contrast], int(z_index))
            axis.set_title(f"MNI z={mni_z:.0f} mm", fontsize=9)
            axis.text(0.02, 0.95, "R", transform=axis.transAxes, color="white", fontsize=9,
                      fontweight="bold", ha="left", va="top")
            axis.text(0.98, 0.95, "L", transform=axis.transAxes, color="white", fontsize=9,
                      fontweight="bold", ha="right", va="top")
            axis.axis("off")
        scalar = ScalarMappable(norm=Normalize(-DISPLAY_Z_MAX, DISPLAY_Z_MAX), cmap="RdBu_r")
        scalar.set_array([])
        cbar = fig.colorbar(
            scalar, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015,
            extend="both", ticks=np.arange(-8, 9, 2),
        )
        cbar.set_label(f"Z statistic (display threshold |Z| >= {DISPLAY_Z_THRESHOLD:g})")
        fig.suptitle(
            f"{DISPLAY_NAMES[contrast]}: Z-map montage\n"
            "Radiological convention (image left = participant right)",
            fontsize=14, y=0.985,
        )
        fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.91, wspace=0.03, hspace=0.13)
        save_publication_figure(fig, figures, f"zmap_montage_{contrast}")
        plt.close(fig)

        thresholded = np.where(mask & (np.abs(data) >= DISPLAY_Z_THRESHOLD), data, 0)
        projections = [signed_mip(thresholded, axis) for axis in (0, 1, 2)]
        affine = reference_image.affine
        x_coords = affine[0, 0] * np.arange(data.shape[0]) + affine[0, 3]
        y_coords = affine[1, 1] * np.arange(data.shape[1]) + affine[1, 3]
        z_coords = affine[2, 2] * np.arange(data.shape[2]) + affine[2, 3]
        displays = [
            (projections[0].T, [y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
             "Sagittal MIP", "MNI y (mm)", "MNI z (mm)", False),
            (projections[1].T, [x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
             "Coronal MIP", "MNI x (mm)", "MNI z (mm)", True),
            (projections[2].T, [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
             "Axial MIP", "MNI x (mm)", "MNI y (mm)", True),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
        for axis, (projection, extent, title, xlabel, ylabel, mark_lr) in zip(axes, displays):
            axis.imshow(
                projection, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r",
                vmin=-DISPLAY_Z_MAX, vmax=DISPLAY_Z_MAX, interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if mark_lr:
                axis.text(0.02, 0.96, "R", transform=axis.transAxes, ha="left", va="top",
                          color="black", fontweight="bold")
                axis.text(0.98, 0.96, "L", transform=axis.transAxes, ha="right", va="top",
                          color="black", fontweight="bold")
        scalar = ScalarMappable(norm=Normalize(-DISPLAY_Z_MAX, DISPLAY_Z_MAX), cmap="RdBu_r")
        scalar.set_array([])
        cbar = fig.colorbar(
            scalar, ax=axes.ravel().tolist(), fraction=0.026, pad=0.025,
            extend="both", ticks=np.arange(-8, 9, 2),
        )
        cbar.set_label(f"Z statistic (display threshold |Z| >= {DISPLAY_Z_THRESHOLD:g})")
        fig.suptitle(f"{DISPLAY_NAMES[contrast]}: whole-brain maximum-intensity projections")
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.14, top=0.84, wspace=0.33)
        save_publication_figure(fig, figures, f"zmap_mip_{contrast}")
        plt.close(fig)


def inflated_surface_figures(
    out: Path, zmaps: dict[str, Path], mask_path: Path, figures: Path,
) -> None:
    fsaverage = nilearn_datasets.fetch_surf_fsaverage("fsaverage5")
    surface_dir = out / "qc" / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    textures: dict[tuple[str, str], np.ndarray] = {}
    missing_fractions: dict[str, float] = {}
    for contrast in PRIMARY:
        for hemisphere in ("left", "right"):
            texture = nilearn_surface.vol_to_surf(
                str(zmaps[contrast]),
                fsaverage[f"pial_{hemisphere}"],
                inner_mesh=fsaverage[f"white_{hemisphere}"],
                mask_img=str(mask_path),
                kind="depth",
                depth=[0.0, 0.25, 0.5, 0.75, 1.0],
                interpolation="linear",
            )
            missing_fraction = float(np.mean(~np.isfinite(texture)))
            if missing_fraction > 0.02:
                raise RuntimeError(
                    f"Excessive missing surface sampling: {contrast} {hemisphere} "
                    f"({100*missing_fraction:.2f}%)"
                )
            texture = np.nan_to_num(texture, nan=0.0, posinf=0.0, neginf=0.0)
            textures[(contrast, hemisphere)] = texture
            missing_fractions[f"{contrast}_{hemisphere}"] = missing_fraction
            gifti = nib.gifti.GiftiImage(
                darrays=[nib.gifti.GiftiDataArray(texture.astype(np.float32))]
            )
            nib.save(gifti, str(surface_dir / f"sigma3_{contrast}_fsaverage5_{hemisphere}.func.gii"))

    views = [
        ("left", "lateral", "L lateral"),
        ("left", "medial", "L medial"),
        ("right", "medial", "R medial"),
        ("right", "lateral", "R lateral"),
    ]

    def draw_surface(axis, contrast: str, hemisphere: str, view: str) -> None:
        nilearn_plotting.plot_surf_stat_map(
            fsaverage[f"infl_{hemisphere}"],
            textures[(contrast, hemisphere)],
            bg_map=fsaverage[f"sulc_{hemisphere}"],
            hemi=hemisphere,
            view=view,
            threshold=DISPLAY_Z_THRESHOLD,
            cmap="RdBu_r",
            vmin=-DISPLAY_Z_MAX,
            vmax=DISPLAY_Z_MAX,
            symmetric_cbar=True,
            colorbar=False,
            bg_on_data=False,
            alpha=0.95,
            axes=axis,
            figure=axis.figure,
        )
        for collection in axis.collections:
            collection.set_rasterized(True)

    fig = plt.figure(figsize=(14.8, 10.2), facecolor="white")
    for row, contrast in enumerate(PRIMARY):
        for column, (hemisphere, view, view_label) in enumerate(views):
            axis = fig.add_subplot(3, 4, row * 4 + column + 1, projection="3d")
            draw_surface(axis, contrast, hemisphere, view)
            if row == 0:
                axis.set_title(view_label, fontsize=11, pad=-2)
            if column == 0:
                axis.text2D(
                    -0.17, 0.5, DISPLAY_NAMES[contrast], transform=axis.transAxes,
                    rotation=90, ha="center", va="center", fontsize=11, fontweight="bold",
                )
    scalar = ScalarMappable(norm=Normalize(-DISPLAY_Z_MAX, DISPLAY_Z_MAX), cmap="RdBu_r")
    scalar.set_array([])
    color_axis = fig.add_axes([0.26, 0.035, 0.48, 0.025])
    colorbar = fig.colorbar(
        scalar, cax=color_axis, orientation="horizontal", extend="both",
        ticks=np.arange(-8, 9, 2),
    )
    colorbar.set_label(f"Z statistic (display threshold |Z| >= {DISPLAY_Z_THRESHOLD:g})")
    fig.suptitle(
        "Cortical projections of the three primary Z maps on fsaverage5 inflated surfaces",
        fontsize=15, y=0.985,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.09, wspace=-0.08, hspace=-0.10)
    save_publication_figure(fig, figures, "zmap_inflated_surface_all", tight=False)
    plt.close(fig)

    for contrast in PRIMARY:
        fig = plt.figure(figsize=(13.5, 4.2), facecolor="white")
        for column, (hemisphere, view, view_label) in enumerate(views):
            axis = fig.add_subplot(1, 4, column + 1, projection="3d")
            draw_surface(axis, contrast, hemisphere, view)
            axis.set_title(view_label, fontsize=10, pad=-2)
        scalar = ScalarMappable(norm=Normalize(-DISPLAY_Z_MAX, DISPLAY_Z_MAX), cmap="RdBu_r")
        scalar.set_array([])
        color_axis = fig.add_axes([0.28, 0.07, 0.44, 0.035])
        colorbar = fig.colorbar(
            scalar, cax=color_axis, orientation="horizontal", extend="both",
            ticks=np.arange(-8, 9, 2),
        )
        colorbar.set_label(f"Z statistic (display threshold |Z| >= {DISPLAY_Z_THRESHOLD:g})")
        fig.suptitle(
            f"{DISPLAY_NAMES[contrast]}: fsaverage5 inflated-surface projection",
            fontsize=14, y=0.965,
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.17, wspace=-0.04)
        save_publication_figure(fig, figures, f"zmap_inflated_surface_{contrast}", tight=False)
        plt.close(fig)

    metadata = {
        "surface_template": "fsaverage5",
        "volume_to_surface_sampling": "linear interpolation at depths 0, 0.25, 0.5, 0.75, and 1 from white to pial",
        "display_threshold_abs_z": DISPLAY_Z_THRESHOLD,
        "display_range_signed_z": [-DISPLAY_Z_MAX, DISPLAY_Z_MAX],
        "interpretation": "cortical visualization only; deep structures remain represented in montages and MIPs",
        "input_maps": {contrast: str(zmaps[contrast]) for contrast in PRIMARY},
        "missing_vertex_fractions_before_zero_fill": missing_fractions,
    }
    (surface_dir / "surface_projection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def save_like(data: np.ndarray, reference: Path, output: Path, dtype=np.uint8) -> None:
    image = nib.load(str(reference))
    header = image.header.copy(); header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data.astype(dtype), image.affine, header), str(output))


def overlap_and_coverage(
    work: Path, out: Path, mask: np.ndarray, atlas: np.ndarray, labels: dict[int, str],
    main_regions: pd.DataFrame, figures: Path, tables: Path,
) -> None:
    template = load(work / "fslvbm/stats/template_GM.nii.gz")
    tfce = pd.read_csv(out / "tables/tfce_by_aal3_region.tsv", sep="\t")
    overlap_rows: list[dict[str, object]] = []
    for contrast_index, contrast in enumerate(PRIMARY, start=1):
        selected = set(main_regions[
            main_regions.contrast.eq(contrast) & main_regions.method.eq("seft") & main_regions.significant_recomputed.eq(1)
        ].region_id.astype(int))
        seft_mask = mask & np.isin(atlas, list(selected))
        corrp_path = out / "randomise" / f"three_group_sigma3_tfce_corrp_tstat{contrast_index}.nii.gz"
        corrp = load(corrp_path)
        tfce_mask = mask & (corrp >= 0.95)
        code = seft_mask.astype(np.uint8) + 2 * tfce_mask.astype(np.uint8)
        output_map = out / "qc" / f"seft_tfce_overlap_{contrast}.nii.gz"
        save_like(code, work / "fslvbm/stats/GM_mask.nii.gz", output_map)
        overlap_rows.append({
            "contrast": contrast, "seft_region_voxels": int(seft_mask.sum()),
            "tfce_significant_voxels": int(tfce_mask.sum()), "overlap_voxels": int((seft_mask & tfce_mask).sum()),
            "tfce_voxels_inside_seft_regions_fraction": float((seft_mask & tfce_mask).sum() / max(tfce_mask.sum(), 1)),
            "overlap_map": str(output_map),
        })
        slices = np.linspace(np.nonzero(mask)[2].min() + 5, np.nonzero(mask)[2].max() - 5, 12).astype(int)
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        cmap = ListedColormap([(0, 0, 0, 0), (0.95, 0.35, 0.12, 0.38), (0.05, 0.45, 0.82, 0.88), (0.55, 0.12, 0.62, 0.95)])
        for axis, z_index in zip(axes.flat, slices):
            bg = np.rot90(template[:, :, z_index])
            low, high = np.quantile(bg[bg > 0], [0.02, 0.98]) if np.any(bg > 0) else (0, 1)
            axis.imshow(bg, cmap="gray", vmin=low, vmax=high)
            axis.imshow(np.rot90(np.where(code[:, :, z_index] > 0, code[:, :, z_index], np.nan)), cmap=cmap, vmin=0, vmax=3)
            mni_z = axial_mni_z(work / "fslvbm/stats/GM_mask.nii.gz", int(z_index))
            axis.set_title(f"MNI z={mni_z:.0f} mm", fontsize=8); axis.axis("off")
        fig.legend(handles=[Patch(color="#F05A24", alpha=.55, label="SEFT-selected AAL3 region"),
                            Patch(color="#0D73C9", alpha=.85, label="TFCE FWER-significant voxel"),
                            Patch(color="#8C1F9F", alpha=.9, label="spatial overlap")],
                   loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"Two-sided SEFT regions and positive-tail TFCE: {DISPLAY_NAMES[contrast]}")
        fig.tight_layout(rect=(0, .04, 1, .97))
        fig.savefig(figures / f"seft_tfce_overlap_{contrast}.png", dpi=220, facecolor="white")
        plt.close(fig)

        coverage = tfce[(tfce.variant.eq("sigma3")) & tfce.contrast.eq(contrast)].copy()
        coverage = coverage[coverage.tfce_coverage_fraction >= 0.01].sort_values("tfce_coverage_fraction", ascending=True)
        coverage["seft_selected"] = coverage.region_id.astype(int).isin(selected)
        coverage.to_csv(tables / f"coverage_plot_data_{contrast}.tsv", sep="\t", index=False)
        if coverage.empty:
            continue
        height = max(4.5, 0.29 * len(coverage))
        fig, (ax_size, ax_cov) = plt.subplots(1, 2, figsize=(14, height), sharey=True,
                                             gridspec_kw={"width_ratios": [1.05, 1.45], "wspace": 0.04})
        y = np.arange(len(coverage))
        ax_size.barh(y, coverage.region_voxels_in_gm_mask, color="#C6DBEF", edgecolor="none")
        ax_size.set(xlabel="AAL3 region size (GM-mask voxels)")
        ax_size.set_yticks(y, coverage.region_label.str.replace("_", " "), fontsize=8)
        ax_size.grid(axis="x", color="0.9", linewidth=.6)
        colors = np.where(coverage.seft_selected, "#F05A28", "#2171B5")
        ax_cov.barh(y, 100 * coverage.tfce_coverage_fraction, color=colors, edgecolor="none")
        ax_cov.set(xlabel="TFCE-significant coverage (%)")
        ax_cov.grid(axis="x", color="0.9", linewidth=.6)
        ax_cov.legend(handles=[Patch(color="#2171B5", label="TFCE coverage; not selected by SEFT"),
                               Patch(color="#F05A28", label="TFCE coverage; selected by SEFT")],
                      frameon=False, loc="lower right", fontsize=8)
        fig.suptitle(f"AAL3 regions with >=1% positive-tail TFCE coverage: {DISPLAY_NAMES[contrast]}\n"
                     "Regions ordered by TFCE coverage proportion")
        fig.savefig(figures / f"region_tfce_coverage_{contrast}.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    pd.DataFrame(overlap_rows).to_csv(tables / "seft_tfce_overlap_summary.tsv", sep="\t", index=False)


def regional_method_and_contrast_overlap(
    work: Path, out: Path, mask: np.ndarray, atlas: np.ndarray, labels: dict[int, str],
    region_long: pd.DataFrame, figures: Path, tables: Path,
) -> None:
    """Compare SEFT with Simes--BHo and quantify nesting across clinical contrasts."""
    contrasts = PRIMARY
    contrast_pairs = [
        ("CN_gt_Dementia", "MCI_gt_Dementia"),
        ("CN_gt_Dementia", "CN_gt_MCI"),
        ("MCI_gt_Dementia", "CN_gt_MCI"),
    ]
    data = region_long[
        region_long.atlas.eq("aal3") & region_long.variant.eq("sigma3")
        & region_long.alpha.eq(0.1) & region_long.contrast.isin(contrasts)
        & region_long.pc_level.isin([0.1, 0.2, 0.3])
        & region_long.method.isin(["seft", "simes"])
    ].copy()
    if data.empty:
        raise RuntimeError("No sigma=3 AAL3 regional results for overlap analysis")

    stability = pd.read_csv(
        work / "results/stability_primary/tables/region_stability_summary.tsv", sep="\t"
    )[[
        "contrast", "region_id", "bootstrap_selection_frequency",
        "delete10_selection_frequency", "stability_class",
    ]]
    tfce = pd.read_csv(out / "tables/tfce_by_aal3_region.tsv", sep="\t")
    tfce = tfce[
        tfce.variant.eq("sigma3") & tfce.contrast.isin(contrasts)
    ][[
        "contrast", "region_id", "tfce_significant_voxels_fwer_0p05",
        "tfce_coverage_fraction",
    ]]

    all_region_ids = sorted(int(value) for value in np.unique(atlas[mask]) if value > 0)
    set_lookup: dict[tuple[str, float, str], set[int]] = {}
    for contrast in contrasts:
        for pc_level in (0.1, 0.2, 0.3):
            for method in ("seft", "simes"):
                block = data[
                    data.contrast.eq(contrast) & np.isclose(data.pc_level, pc_level)
                    & data.method.eq(method) & data.significant_recomputed.eq(1)
                ]
                set_lookup[(contrast, pc_level, method)] = set(block.region_id.astype(int))

    summary_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    for contrast in contrasts:
        for pc_level in (0.1, 0.2, 0.3):
            seft = set_lookup[(contrast, pc_level, "seft")]
            simes = set_lookup[(contrast, pc_level, "simes")]
            shared, seft_only, simes_only = seft & simes, seft - simes, simes - seft
            union = seft | simes
            summary_rows.append({
                "contrast": contrast, "pc_level": pc_level,
                "seft_discoveries": len(seft), "simes_bh_discoveries": len(simes),
                "shared_discoveries": len(shared), "seft_only": len(seft_only),
                "simes_bh_only": len(simes_only), "union_discoveries": len(union),
                "jaccard": len(shared) / len(union) if union else 1.0,
                "fraction_seft_shared": len(shared) / len(seft) if seft else np.nan,
                "fraction_simes_bh_shared": len(shared) / len(simes) if simes else np.nan,
            })
            for region_id in all_region_ids:
                in_seft, in_simes = region_id in seft, region_id in simes
                category = (
                    "shared" if in_seft and in_simes else
                    "seft_only" if in_seft else
                    "simes_bh_only" if in_simes else "neither"
                )
                region_rows.append({
                    "contrast": contrast, "pc_level": pc_level,
                    "region_id": region_id,
                    "region_label": labels.get(region_id, f"region_{region_id}"),
                    "seft_selected": int(in_seft), "simes_bh_selected": int(in_simes),
                    "method_overlap_category": category,
                })
    summary = pd.DataFrame(summary_rows)
    by_region = pd.DataFrame(region_rows).merge(
        stability, on=["contrast", "region_id"], how="left", validate="many_to_one"
    ).merge(tfce, on=["contrast", "region_id"], how="left", validate="many_to_one")
    summary.to_csv(tables / "seft_simes_overlap_summary.tsv", sep="\t", index=False)
    by_region.to_csv(tables / "seft_simes_overlap_by_region.tsv", sep="\t", index=False)

    contrast_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for method in ("seft", "simes"):
        for pc_level in (0.1, 0.2, 0.3):
            for left, right in contrast_pairs:
                left_set = set_lookup[(left, pc_level, method)]
                right_set = set_lookup[(right, pc_level, method)]
                intersection = left_set & right_set
                union = left_set | right_set
                contrast_rows.append({
                    "method": "simes_bh" if method == "simes" else method,
                    "pc_level": pc_level, "contrast_a": left, "contrast_b": right,
                    "discoveries_a": len(left_set), "discoveries_b": len(right_set),
                    "intersection": len(intersection), "union": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                    "fraction_a_in_b": len(intersection) / len(left_set) if left_set else np.nan,
                    "fraction_b_in_a": len(intersection) / len(right_set) if right_set else np.nan,
                })
            for region_id in all_region_ids:
                flags = [region_id in set_lookup[(contrast, pc_level, method)] for contrast in contrasts]
                membership_rows.append({
                    "method": "simes_bh" if method == "simes" else method,
                    "pc_level": pc_level, "region_id": region_id,
                    "region_label": labels.get(region_id, f"region_{region_id}"),
                    **{f"selected_{contrast}": int(flag) for contrast, flag in zip(contrasts, flags)},
                    "n_contrasts_selected": int(sum(flags)),
                    "membership_pattern": "|".join(
                        contrast for contrast, flag in zip(contrasts, flags) if flag
                    ) or "none",
                })
    pd.DataFrame(contrast_rows).to_csv(
        tables / "contrast_region_overlap_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(membership_rows).to_csv(
        tables / "contrast_region_membership.tsv", sep="\t", index=False
    )

    # Discovery-set overlap counts across PC fractions.
    plot = summary.copy()
    plot["label"] = plot.contrast.map(DISPLAY_NAMES) + plot.pc_level.map(
        {0.1: "  (c=0.10)", 0.2: "  (c=0.20)", 0.3: "  (c=0.30)"}
    )
    plot["order"] = plot.pc_level.map({0.1: 0, 0.2: 1, 0.3: 2}) * 3 + plot.contrast.map(
        {name: index for index, name in enumerate(contrasts)}
    )
    plot = plot.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    y = np.arange(len(plot))
    left = np.zeros(len(plot))
    for column, color, label in (
        ("shared_discoveries", "#7A5195", "Selected by both"),
        ("seft_only", "#F05A28", "SEFT only"),
        ("simes_bh_only", "#2B6CB0", "Simes--BHo only"),
    ):
        values = plot[column].to_numpy(float)
        ax.barh(y, values, left=left, color=color, label=label)
        for index, (start, value) in enumerate(zip(left, values)):
            if value >= 4:
                ax.text(start + value / 2, index, f"{int(value)}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        left += values
    ax.set_yticks(y, plot.label)
    ax.set_xlabel("Number of AAL3 regions in the union")
    ax.set_title("SEFT and Simes--BHo overlap for identical regional PC hypotheses")
    ax.grid(axis="x", color="0.9", linewidth=.6)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "seft_simes_overlap_counts.png", dpi=220, facecolor="white")
    plt.close(fig)

    template = load(work / "fslvbm/stats/template_GM.nii.gz")
    slices = np.linspace(np.nonzero(mask)[2].min() + 7, np.nonzero(mask)[2].max() - 7, 6).astype(int)
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
    cmap = ListedColormap([(0, 0, 0, 0), (0.48, 0.32, 0.66, .88),
                           (0.94, 0.31, 0.12, .88), (0.10, 0.40, 0.72, .88)])
    for row_index, contrast in enumerate(contrasts):
        seft = set_lookup[(contrast, 0.2, "seft")]
        simes = set_lookup[(contrast, 0.2, "simes")]
        code = np.zeros(mask.shape, dtype=np.uint8)
        code[mask & np.isin(atlas, list(seft & simes))] = 1
        code[mask & np.isin(atlas, list(seft - simes))] = 2
        code[mask & np.isin(atlas, list(simes - seft))] = 3
        save_like(code, work / "fslvbm/stats/GM_mask.nii.gz",
                  out / "qc" / f"seft_simes_overlap_{contrast}_c0p2.nii.gz")
        for column_index, z_index in enumerate(slices):
            axis = axes[row_index, column_index]
            bg = np.rot90(template[:, :, z_index])
            low, high = np.quantile(bg[bg > 0], [0.02, 0.98]) if np.any(bg > 0) else (0, 1)
            axis.imshow(bg, cmap="gray", vmin=low, vmax=high)
            axis.imshow(np.rot90(np.where(code[:, :, z_index] > 0, code[:, :, z_index], np.nan)),
                        cmap=cmap, vmin=0, vmax=3)
            if row_index == 0:
                axis.set_title(f"MNI z={axial_mni_z(work / 'fslvbm/stats/GM_mask.nii.gz', int(z_index)):.0f}",
                               fontsize=8)
            if column_index == 0:
                axis.set_ylabel(DISPLAY_NAMES[contrast], fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
    fig.legend(handles=[Patch(color="#7A5195", label="both"),
                        Patch(color="#F05A28", label="SEFT only"),
                        Patch(color="#2B6CB0", label="Simes--BHo only")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Regional method overlap at c=0.20 and alpha=0.10", y=.99)
    fig.tight_layout(rect=(0, .055, 1, .97))
    fig.savefig(figures / "seft_simes_overlap_montage.png", dpi=220, facecolor="white")
    plt.close(fig)

    # Anatomical nesting of the two non-empty primary SEFT contrasts.
    cn_dem = set_lookup[("CN_gt_Dementia", 0.2, "seft")]
    mci_dem = set_lookup[("MCI_gt_Dementia", 0.2, "seft")]
    code = np.zeros(mask.shape, dtype=np.uint8)
    code[mask & np.isin(atlas, list(cn_dem & mci_dem))] = 1
    code[mask & np.isin(atlas, list(cn_dem - mci_dem))] = 2
    code[mask & np.isin(atlas, list(mci_dem - cn_dem))] = 3
    save_like(code, work / "fslvbm/stats/GM_mask.nii.gz",
              out / "qc" / "seft_contrast_overlap_c0p2.nii.gz")
    slices12 = np.linspace(np.nonzero(mask)[2].min() + 5, np.nonzero(mask)[2].max() - 5, 12).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    cmap2 = ListedColormap([(0, 0, 0, 0), (0.46, 0.30, 0.65, .88),
                            (0.95, 0.45, 0.15, .88), (0.12, 0.55, 0.43, .88)])
    for axis, z_index in zip(axes.flat, slices12):
        bg = np.rot90(template[:, :, z_index])
        low, high = np.quantile(bg[bg > 0], [0.02, 0.98]) if np.any(bg > 0) else (0, 1)
        axis.imshow(bg, cmap="gray", vmin=low, vmax=high)
        axis.imshow(np.rot90(np.where(code[:, :, z_index] > 0, code[:, :, z_index], np.nan)),
                    cmap=cmap2, vmin=0, vmax=3)
        axis.set_title(f"MNI z={axial_mni_z(work / 'fslvbm/stats/GM_mask.nii.gz', int(z_index)):.0f} mm",
                       fontsize=8)
        axis.axis("off")
    fig.legend(handles=[Patch(color="#7550A3", label="selected in both contrasts"),
                        Patch(color="#F27327", label="CN--dementia only"),
                        Patch(color="#1F8C6B", label="MCI--dementia only")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("SEFT anatomical overlap across primary clinical contrasts (c=0.20)")
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.savefig(figures / "seft_contrast_overlap_montage.png", dpi=220, facecolor="white")
    plt.close(fig)


def regional_direction_summary(
    zmaps: dict[str, Path], mask: np.ndarray, atlas: np.ndarray, labels: dict[int, str],
    main_regions: pd.DataFrame, tables: Path,
) -> None:
    """Summarize signed effects separately from the two-sided PC decisions."""
    rows: list[dict[str, object]] = []
    for contrast in PRIMARY:
        z_data = load(zmaps[contrast], dtype=np.float64)
        selected = set(main_regions[
            main_regions.contrast.eq(contrast) & main_regions.method.eq("seft")
            & main_regions.significant_recomputed.eq(1)
        ].region_id.astype(int))
        for region_id in sorted(int(value) for value in np.unique(atlas[mask]) if value > 0):
            values = finite_values(z_data, mask & atlas.__eq__(region_id))
            if values.size == 0:
                continue
            mean_z = float(np.mean(values))
            median_z = float(np.median(values))
            rows.append({
                "contrast": contrast,
                "regional_pc_test_sidedness": "two-sided",
                "region_id": region_id,
                "region_label": labels.get(region_id, f"region_{region_id}"),
                "n_voxels": int(values.size),
                "mean_signed_z": mean_z,
                "median_signed_z": median_z,
                "positive_z_fraction": float(np.mean(values > 0)),
                "predominant_direction": "positive" if mean_z > 0 else ("negative" if mean_z < 0 else "neutral"),
                "seft_selected_alpha0p1_c0p2": int(region_id in selected),
            })
    pd.DataFrame(rows).to_csv(tables / "regional_signed_direction_summary.tsv", sep="\t", index=False)


def cn_mci_c0p1_overlap(
    work: Path, out: Path, mask: np.ndarray, atlas: np.ndarray,
    region_long: pd.DataFrame, figures: Path, tables: Path,
) -> None:
    selected_rows = region_long[
        region_long.atlas.eq("aal3") & region_long.variant.eq("sigma3")
        & region_long.contrast.eq("CN_gt_MCI") & region_long.method.eq("seft")
        & np.isclose(region_long.pc_level, 0.1) & np.isclose(region_long.alpha, 0.1)
        & region_long.significant_recomputed.eq(1)
    ]
    selected = set(selected_rows.region_id.astype(int))
    seft_mask = mask & np.isin(atlas, list(selected))
    corrp_path = out / "randomise/three_group_sigma3_tfce_corrp_tstat3.nii.gz"
    tfce_mask = mask & (load(corrp_path) >= 0.95)
    code = seft_mask.astype(np.uint8) + 2 * tfce_mask.astype(np.uint8)
    output_map = out / "qc/seft_tfce_overlap_CN_gt_MCI_c0p1.nii.gz"
    save_like(code, work / "fslvbm/stats/GM_mask.nii.gz", output_map)
    pd.DataFrame([{
        "contrast": "CN_gt_MCI", "pc_level": 0.1,
        "selected_regions": len(selected), "seft_region_voxels": int(seft_mask.sum()),
        "tfce_significant_voxels": int(tfce_mask.sum()),
        "overlap_voxels": int((seft_mask & tfce_mask).sum()),
        "tfce_voxels_inside_seft_regions_fraction": float(
            (seft_mask & tfce_mask).sum() / max(tfce_mask.sum(), 1)
        ),
        "overlap_map": str(output_map),
    }]).to_csv(tables / "cn_mci_c0p1_seft_tfce_overlap.tsv", sep="\t", index=False)

    template = load(work / "fslvbm/stats/template_GM.nii.gz")
    slices = np.linspace(np.nonzero(mask)[2].min() + 5, np.nonzero(mask)[2].max() - 5, 12).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    cmap = ListedColormap([(0, 0, 0, 0), (0.95, 0.35, 0.12, 0.38),
                           (0.05, 0.45, 0.82, 0.88), (0.55, 0.12, 0.62, 0.95)])
    for axis, z_index in zip(axes.flat, slices):
        bg = np.rot90(template[:, :, z_index])
        low, high = np.quantile(bg[bg > 0], [0.02, 0.98]) if np.any(bg > 0) else (0, 1)
        axis.imshow(bg, cmap="gray", vmin=low, vmax=high)
        axis.imshow(np.rot90(np.where(code[:, :, z_index] > 0, code[:, :, z_index], np.nan)),
                    cmap=cmap, vmin=0, vmax=3)
        mni_z = axial_mni_z(work / "fslvbm/stats/GM_mask.nii.gz", int(z_index))
        axis.set_title(f"MNI z={mni_z:.0f} mm", fontsize=8)
        axis.axis("off")
    fig.legend(handles=[Patch(color="#F05A24", alpha=.55, label="SEFT region (c=0.10)"),
                        Patch(color="#0D73C9", alpha=.85, label="TFCE FWER-significant voxel"),
                        Patch(color="#8C1F9F", alpha=.9, label="spatial overlap")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("CN > MCI: c=0.10 SEFT regions and positive-tail TFCE")
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(figures / "seft_tfce_overlap_CN_gt_MCI_c0p1.png", dpi=220, facecolor="white")
    plt.close(fig)


def diagnosis_date_concordance(work: Path, tables: Path) -> None:
    """Audit visit-code linkage, calendar gaps, and nearest dated diagnosis."""
    if DXSUM is None:
        raise RuntimeError("The DXSUM metadata file has not been resolved")
    subjects = pd.read_csv(work / "tables/sample_final_with_tiv.tsv", sep="\t", dtype={"subject_id": str})
    diagnoses = pd.read_csv(DXSUM, dtype=str)
    diagnoses = diagnoses[diagnoses.DIAGNOSIS.isin(["1", "2", "3"])].copy()
    diagnoses["diagnosis_date"] = pd.to_datetime(diagnoses.EXAMDATE, errors="coerce")
    label_lookup = {"1": "CN", "2": "MCI", "3": "Dementia"}
    rows: list[dict[str, object]] = []
    for subject in subjects.itertuples(index=False):
        image_date = pd.to_datetime(subject.image_date)
        linked_date = pd.to_datetime(subject.EXAMDATE, errors="coerce")
        available = diagnoses[
            diagnoses.PTID.eq(subject.subject_id) & diagnoses.diagnosis_date.notna()
        ].copy()
        available["absolute_gap_days"] = (available.diagnosis_date - image_date).dt.days.abs()
        nearest = available.sort_values(["absolute_gap_days", "diagnosis_date", "ID"]).iloc[0]
        nearest_label = label_lookup[str(nearest.DIAGNOSIS)]
        rows.append({
            "triplet_id": subject.triplet_id,
            "label": subject.label,
            "subject_id": subject.subject_id,
            "image_visit": subject.image_visit,
            "image_date": subject.image_date,
            "visit_code_linked_diagnosis_date": subject.EXAMDATE,
            "visit_code_linked_absolute_gap_days": (
                float(abs((linked_date - image_date).days)) if pd.notna(linked_date) else np.nan
            ),
            "nearest_dated_diagnosis_visit": nearest.VISCODE2,
            "nearest_dated_diagnosis_date": nearest.EXAMDATE,
            "nearest_dated_diagnosis_label": nearest_label,
            "nearest_dated_diagnosis_absolute_gap_days": int(nearest.absolute_gap_days),
            "nearest_diagnosis_category_concordant": int(nearest_label == subject.label),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "diagnosis_date_concordance.tsv", sep="\t", index=False)
    linked = frame.visit_code_linked_absolute_gap_days.dropna()
    nearest = frame.nearest_dated_diagnosis_absolute_gap_days
    summary = pd.DataFrame([{
        "n_subjects": len(frame),
        "same_visit_code_linkage_fraction": 1.0,
        "missing_linked_diagnosis_date": int(frame.visit_code_linked_diagnosis_date.isna().sum()),
        "linked_gap_median_days": float(linked.median()),
        "linked_gap_q90_days": float(linked.quantile(0.90)),
        "linked_gap_gt90_n": int((linked > 90).sum()),
        "linked_gap_gt180_n": int((linked > 180).sum()),
        "nearest_gap_median_days": float(nearest.median()),
        "nearest_gap_q90_days": float(nearest.quantile(0.90)),
        "nearest_gap_gt90_n": int((nearest > 90).sum()),
        "nearest_diagnosis_category_concordant_n": int(frame.nearest_diagnosis_category_concordant.sum()),
        "nearest_diagnosis_category_discordant_n": int((1 - frame.nearest_diagnosis_category_concordant).sum()),
    }])
    summary.to_csv(tables / "diagnosis_date_concordance_summary.tsv", sep="\t", index=False)


def sensitivity_figure(out: Path, figures: Path) -> None:
    summary = pd.read_csv(out / "tables/seft_sensitivity_summary.tsv", sep="\t")
    summary["configuration"] = (
        summary.atlas + " | " + summary.variant + " | c=" + summary.pc_level.map(lambda x: f"{x:g}")
        + " | alpha=" + summary.alpha.map(lambda x: f"{x:g}")
    )
    methods = [value for value in ("seft", "simes") if value in set(summary.method)]
    fig, axes = plt.subplots(len(methods), len(PRIMARY), figsize=(16, 5.2 * len(methods)), squeeze=False)
    for row_index, method in enumerate(methods):
        method_data = summary[summary.method.eq(method)]
        configurations = list(dict.fromkeys(method_data.configuration))
        for col_index, contrast in enumerate(PRIMARY):
            block = method_data[method_data.contrast.eq(contrast)].set_index("configuration")
            values = block.reindex(configurations).discovered_regions.to_numpy(float)[:, None]
            axis = axes[row_index, col_index]
            image = axis.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0)
            axis.set_xticks([0], [method.upper()])
            axis.set_yticks(np.arange(len(configurations)), configurations if col_index == 0 else [], fontsize=7)
            axis.set_title(DISPLAY_NAMES[contrast])
            for index, value in enumerate(values[:, 0]):
                if np.isfinite(value):
                    axis.text(0, index, f"{int(value)}", ha="center", va="center", fontsize=7,
                              color="white" if value > np.nanmax(values) * .55 else "black")
            plt.colorbar(image, ax=axis, fraction=0.08, pad=0.04, label="Discovered regions")
    fig.suptitle("Atlas, PC-fraction, alpha, and input-smoothing sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "seft_sensitivity_heatmap.png", dpi=220, facecolor="white")
    plt.close(fig)


def kernel_bandwidth_figure(work: Path, figures: Path) -> None:
    path = work / "results/kernel_smoothing_sensitivity/tables/kernel_bandwidth_sensitivity.tsv"
    if not good(path):
        raise RuntimeError(f"Missing kernel sensitivity table: {path}")
    summary = pd.read_csv(path, sep="\t")
    block = summary[
        np.isclose(summary.pc_level, 0.2) & np.isclose(summary.alpha, 0.1)
    ].copy()
    expected = len(PRIMARY) * 2 * 4
    if len(block) != expected:
        raise RuntimeError(f"Expected {expected} primary kernel sensitivity rows; got {len(block)}")
    colors = {"sigma2": "#2474B7", "sigma3": "#E4572E"}
    labels = {"sigma2": "Input smoothing sigma=2 mm", "sigma3": "Input smoothing sigma=3 mm"}
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(15, 4.8), sharey=True)
    for axis, contrast in zip(axes, PRIMARY):
        for variant in ("sigma2", "sigma3"):
            values = block[block.contrast.eq(contrast) & block.variant.eq(variant)].sort_values(
                "kernel_sigma_mm"
            )
            axis.plot(
                values.kernel_sigma_mm, values.discovered_regions, marker="o", markersize=7,
                linewidth=2.2, color=colors[variant], label=labels[variant],
            )
            for row in values.itertuples():
                axis.annotate(
                    str(int(row.discovered_regions)),
                    (row.kernel_sigma_mm, row.discovered_regions),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                    color=colors[variant],
                )
        axis.axvline(10, color="#111111", linestyle="--", linewidth=1.4, alpha=.9)
        axis.set_xticks([2, 3, 5, 10], ["2", "3", "5", "10\n(fixed default)"])
        axis.set_xlabel("SEFT Gaussian kernel sigma (mm)")
        axis.set_title(DISPLAY_NAMES[contrast])
        axis.grid(axis="y", color="#D9D9D9", linewidth=.8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("SEFT-discovered AAL3 regions\n(c=0.20, alpha=0.10)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("SEFT spatial-kernel sensitivity across two input-smoothing scales", y=1.12, fontsize=15)
    fig.tight_layout()
    fig.savefig(figures / "seft_kernel_bandwidth_sensitivity.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pc_fraction_figure(out: Path, figures: Path) -> None:
    summary = pd.read_csv(out / "tables/seft_sensitivity_summary.tsv", sep="\t")
    block = summary[
        summary.atlas.eq("aal3") & summary.variant.eq("sigma3")
        & summary.method.eq("seft") & np.isclose(summary.alpha, 0.1)
        & summary.contrast.isin(PRIMARY)
    ].copy()
    colors = {
        "CN_gt_Dementia": "#1B9E77", "MCI_gt_Dementia": "#D95F02", "CN_gt_MCI": "#7570B3",
    }
    fig, axis = plt.subplots(figsize=(8.4, 5.4))
    for contrast in PRIMARY:
        values = block[block.contrast.eq(contrast)].sort_values("pc_level")
        axis.plot(values.pc_level, values.discovered_regions, marker="o", markersize=8,
                  linewidth=2.4, color=colors[contrast], label=DISPLAY_NAMES[contrast])
        for row in values.itertuples():
            axis.annotate(str(int(row.discovered_regions)),
                          (row.pc_level, row.discovered_regions), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=9,
                          color=colors[contrast])
    axis.set_xticks([0.1, 0.2, 0.3])
    axis.set(xlabel="Partial-conjunction fraction c",
             ylabel="SEFT-discovered AAL3 regions",
             title="Primary sigma=3 map: sensitivity to the PC fraction")
    axis.grid(color="#DDDDDD", linewidth=.8)
    axis.legend(frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "seft_pc_fraction_sensitivity.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sample_flow(work: Path, figures: Path) -> None:
    selection_path = work / "provenance/cohort_selection.json"
    selection = json.loads(selection_path.read_text())
    final = pd.read_csv(work / "tables/analysis_order.tsv", sep="\t")
    excluded_triplets = 0
    excluded_path = work / "tables/excluded_preprocessing_triplets.tsv"
    if excluded_path.exists():
        excluded_triplets = pd.read_csv(excluded_path, sep="\t").triplet_id.nunique()
    eligible = selection.get("eligible_counts", {})
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.axis("off")
    boxes = [
        (0.5, .88, f"ADNI3 3T T1 scans linked to diagnosis by ADNI visit code\n"
                    f"CN={eligible.get('CN', 0)}, MCI={eligible.get('MCI', 0)}, dementia={eligible.get('Dementia', 0)}"),
        (0.5, .65, "Earliest qualifying visit per subject\nstandard acquisition preferred; age/sex/education complete"),
        (0.5, .42, f"Exact sex + scanner/model/protocol matching\n108 candidate triplets (324 participants)"),
        (0.5, .19, f"BET/FAST/registration QC; complete-triplet exclusion={excluded_triplets}\n"
                    f"Final CN={int((final.label=='CN').sum())}, MCI={int((final.label=='MCI').sum())}, "
                    f"dementia={int((final.label=='Dementia').sum())}"),
    ]
    for x, y, text in boxes:
        axis.text(x, y, text, ha="center", va="center", fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.65", facecolor="#EDF4FB", edgecolor="#477AA1", linewidth=1.5))
    for start, end in ((.80, .73), (.57, .50), (.34, .27)):
        axis.annotate("", xy=(.5, end), xytext=(.5, start), arrowprops=dict(arrowstyle="-|>", color="#477AA1", lw=1.6))
    axis.set_title("ADNI3 application cohort flow", fontsize=16, pad=18)
    fig.savefig(figures / "sample_flow_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global DXSUM, MNI_T1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mni-t1", type=Path, default=MNI_T1)
    args = parser.parse_args()
    DXSUM = resolve_metadata_file(args.data_dir, "dxsum")
    MNI_T1 = args.mni_t1.resolve()
    work = args.work_dir.resolve()
    out = work / "results/application_methods"
    if not good(out / "analysis_complete.ok"):
        raise SystemExit("Primary methods are not complete")
    tables = out / "qc/tables"
    figures = work / "manuscript/figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    mask = load(work / "fslvbm/stats/GM_mask.nii.gz") > 0
    atlas = np.rint(load(AAL3)).astype(int)
    labels = label_lookup()
    zmaps = {contrast: out / "zmaps" / f"sigma3_{contrast}_signed_z.nii.gz" for contrast in PRIMARY}
    for path in zmaps.values():
        if not good(path):
            raise RuntimeError(f"Missing Z map: {path}")
    region_long = pd.read_csv(out / "tables/seft_simes_region_results_long.tsv", sep="\t")
    main_regions = region_long[
        region_long.atlas.eq("aal3") & region_long.variant.eq("sigma3")
        & region_long.pc_level.eq(0.2) & region_long.alpha.eq(0.1)
        & region_long.contrast.isin(PRIMARY)
    ].copy()
    if not good(out / "qc/extended_model_checks_complete.ok"):
        raise RuntimeError(
            "Extended model checks are incomplete; run scripts/run_real_map_extended_checks.R first"
        )
    z_model_check(out, zmaps, mask, atlas, labels, tables, figures)
    claw_diagnostics(out, mask, atlas, labels, main_regions, tables, figures)
    score_cap_sensitivity_figure(tables, figures)
    zmap_figures(zmaps, mask, figures)
    inflated_surface_figures(out, zmaps, work / "fslvbm/stats/GM_mask.nii.gz", figures)
    overlap_and_coverage(work, out, mask, atlas, labels, main_regions, figures, tables)
    regional_method_and_contrast_overlap(
        work, out, mask, atlas, labels, region_long, figures, tables
    )
    cn_mci_c0p1_overlap(work, out, mask, atlas, region_long, figures, tables)
    regional_direction_summary(zmaps, mask, atlas, labels, main_regions, tables)
    diagnosis_date_concordance(work, tables)
    sensitivity_figure(out, figures)
    pc_fraction_figure(out, figures)
    kernel_bandwidth_figure(work, figures)
    sample_flow(work, figures)
    manifest = []
    for path in sorted(figures.glob("*")):
        if path.suffix.lower() not in {".png", ".pdf"}:
            continue
        manifest.append({
            "figure": path.stem, "format": path.suffix.lower().lstrip("."),
            "path": str(path), "size_bytes": path.stat().st_size,
        })
    pd.DataFrame(manifest).to_csv(work / "manuscript/figure_manifest.tsv", sep="\t", index=False)
    (out / "qc/application_qc_complete.ok").write_text(pd.Timestamp.now(tz="Asia/Shanghai").isoformat() + "\n")
    print(f"Generated {len(manifest)} manuscript figures")


if __name__ == "__main__":
    main()
