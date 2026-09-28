#!/usr/bin/env python3
"""Generate compact combined slice-montage and MIP figures for the SEFT paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from nilearn import datasets as nilearn_datasets
from nilearn import plotting as nilearn_plotting


REPRO_ROOT = Path(__file__).resolve().parents[3]
WORK = REPRO_ROOT / "outputs/adni3"
MNI_T1 = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "data/standard/MNI152_T1_2mm_brain.nii.gz"
CONTRASTS = ("CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI")
DISPLAY_NAMES = {
    "CN_gt_Dementia": "CN > dementia",
    "MCI_gt_Dementia": "MCI > dementia",
    "CN_gt_MCI": "CN > MCI",
}
Z_THRESHOLD = 2.0
Z_MAX = 8.0


def load(path: Path, dtype=np.float32) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def signed_mip(data: np.ndarray, axis: int) -> np.ndarray:
    index = np.argmax(np.abs(data), axis=axis)
    return np.take_along_axis(
        data, np.expand_dims(index, axis=axis), axis=axis
    ).squeeze(axis)


def mni_z(reference: nib.spatialimages.SpatialImage, index: int) -> float:
    point = np.asarray(
        [(size - 1) / 2 for size in reference.shape[:3]] + [1.0],
        dtype=float,
    )
    point[2] = index
    return float((reference.affine @ point)[2])


def save(fig: plt.Figure, output: Path, stem: str, *, tight: bool = False) -> None:
    options = {"dpi": 400, "facecolor": "white"}
    if tight:
        options.update(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / f"{stem}.pdf", **options)


def common_colorbar(
    fig: plt.Figure,
    position: list[float],
    *,
    label: str = "Z statistic (|Z| ≥ 2)",
    label_fontsize: float = 10,
    tick_fontsize: float = 8.5,
    orientation: str = "horizontal",
) -> None:
    color_axis = fig.add_axes(position)
    scalar = ScalarMappable(
        norm=Normalize(-Z_MAX, Z_MAX), cmap="RdBu_r"
    )
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        cax=color_axis,
        orientation=orientation,
        extend="both",
        ticks=np.arange(-8, 9, 2),
    )
    colorbar.set_label(label, fontsize=label_fontsize, labelpad=3)
    colorbar.ax.tick_params(labelsize=tick_fontsize, length=2.5, pad=2)
    colorbar.outline.set_linewidth(0.6)


def combined_slice_montage(
    maps: dict[str, np.ndarray],
    references: dict[str, nib.spatialimages.SpatialImage],
    mask: np.ndarray,
    background: np.ndarray,
    output: Path,
) -> None:
    active_z = np.flatnonzero(np.any(mask, axis=(0, 1)))
    slice_indices = np.unique(
        np.rint(np.linspace(active_z.min() + 3, active_z.max() - 3, 10)).astype(int)
    )
    if slice_indices.size != 10:
        raise RuntimeError("Could not select 10 distinct axial slices")

    background_values = background[background > 0]
    bg_low, bg_high = np.quantile(background_values, [0.02, 0.99])
    fig = plt.figure(figsize=(10.6, 3.9))
    grid = fig.add_gridspec(
        3,
        10,
        left=0.060,
        right=0.925,
        bottom=0.025,
        top=0.985,
        wspace=0.015,
        hspace=0.05,
    )

    for contrast_index, contrast in enumerate(CONTRASTS):
        data = maps[contrast]
        for panel_index, z_index in enumerate(slice_indices):
            axis = fig.add_subplot(grid[contrast_index, panel_index])
            background_plane = np.rot90(background[:, :, z_index])
            overlay = np.rot90(
                np.where(
                    mask[:, :, z_index]
                    & (np.abs(data[:, :, z_index]) >= Z_THRESHOLD),
                    data[:, :, z_index],
                    np.nan,
                )
            )
            axis.imshow(
                background_plane,
                cmap="gray",
                vmin=bg_low,
                vmax=bg_high,
                interpolation="nearest",
            )
            axis.imshow(
                overlay,
                cmap="RdBu_r",
                vmin=-Z_MAX,
                vmax=Z_MAX,
                interpolation="nearest",
                alpha=0.88,
            )
            axis.text(
                0.50,
                0.985,
                f"z={mni_z(references[contrast], int(z_index)):+.0f}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                color="white",
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.60,
                    "pad": 1.0,
                },
            )
            if panel_index == 0:
                axis.text(
                    0.025,
                    0.08,
                    "R",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10.5,
                    color="white",
                    fontweight="bold",
                )
                axis.text(
                    0.975,
                    0.08,
                    "L",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10.5,
                    color="white",
                    fontweight="bold",
                )
            axis.axis("off")

        group_position = grid[contrast_index, 0].get_position(fig)
        fig.text(
            0.038,
            (group_position.y0 + group_position.y1) / 2,
            DISPLAY_NAMES[contrast],
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
        )

    common_colorbar(
        fig,
        [0.941, 0.20, 0.012, 0.60],
        label="",
        label_fontsize=10,
        tick_fontsize=9,
        orientation="vertical",
    )
    save(fig, output, "zmap_montage_all", tight=True)
    plt.close(fig)


def combined_mip(
    maps: dict[str, np.ndarray],
    reference: nib.spatialimages.SpatialImage,
    mask: np.ndarray,
    output: Path,
) -> None:
    affine = reference.affine
    x = affine[0, 0] * np.arange(reference.shape[0]) + affine[0, 3]
    y = affine[1, 1] * np.arange(reference.shape[1]) + affine[1, 3]
    z = affine[2, 2] * np.arange(reference.shape[2]) + affine[2, 3]
    projection_specs = (
        (0, [y[0], y[-1], z[0], z[-1]], "Sagittal", "MNI y (mm)", "MNI z (mm)", False),
        (1, [x[0], x[-1], z[0], z[-1]], "Coronal", "MNI x (mm)", "MNI z (mm)", True),
        (2, [x[0], x[-1], y[0], y[-1]], "Axial", "MNI x (mm)", "MNI y (mm)", True),
    )

    fig = plt.figure(figsize=(8.8, 7.7))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.995,
        bottom=0.120,
        top=0.97,
        wspace=0.12,
        hspace=0.22,
    )
    for row, contrast in enumerate(CONTRASTS):
        thresholded = np.where(
            mask & (np.abs(maps[contrast]) >= Z_THRESHOLD), maps[contrast], 0
        )
        for column, (
            projection_axis,
            extent,
            title,
            xlabel,
            ylabel,
            mark_lr,
        ) in enumerate(projection_specs):
            axis = fig.add_subplot(grid[row, column])
            projection = signed_mip(thresholded, projection_axis).T
            brain_outline = np.any(mask, axis=projection_axis).T
            masked_projection = np.ma.masked_where(projection == 0, projection)
            axis.set_facecolor("#F2F2F2")
            axis.imshow(
                masked_projection,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="RdBu_r",
                vmin=-Z_MAX,
                vmax=Z_MAX,
                interpolation="nearest",
            )
            axis.contour(
                brain_outline.astype(float),
                levels=[0.5],
                colors=["0.45"],
                linewidths=0.55,
                origin="lower",
                extent=extent,
            )
            if row == 0:
                axis.set_title(title, fontsize=12, fontweight="semibold", pad=5)
            axis.set_xlabel(xlabel, fontsize=9.5, labelpad=2)
            axis.set_ylabel(ylabel, fontsize=9.5, labelpad=2)
            axis.tick_params(labelsize=8.3, length=2.5, pad=2)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.35")
            if mark_lr:
                axis.text(
                    0.025,
                    0.965,
                    "R",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                )
                axis.text(
                    0.975,
                    0.965,
                    "L",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                )
        row_position = grid[row, 0].get_position(fig)
        fig.text(
            0.020,
            (row_position.y0 + row_position.y1) / 2,
            DISPLAY_NAMES[contrast],
            rotation=90,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="semibold",
        )

    common_colorbar(fig, [0.32, 0.045, 0.42, 0.019])
    save(fig, output, "zmap_mip_all")
    plt.close(fig)


def compact_surface_figure(
    maps: dict[str, np.ndarray],
    references: dict[str, nib.spatialimages.SpatialImage],
    output: Path,
) -> None:
    fsaverage = nilearn_datasets.fetch_surf_fsaverage("fsaverage5")
    surface_dir = WORK / "results/application_methods/qc/surfaces"
    textures: dict[tuple[str, str], np.ndarray] = {}
    for contrast in CONTRASTS:
        for hemisphere in ("left", "right"):
            path = (
                surface_dir
                / f"sigma3_{contrast}_fsaverage5_{hemisphere}.func.gii"
            )
            textures[(contrast, hemisphere)] = np.asarray(
                nib.load(str(path)).darrays[0].data, dtype=np.float32
            )

    views = (
        ("left", "lateral", "L lat."),
        ("left", "medial", "L med."),
        ("right", "medial", "R med."),
        ("right", "lateral", "R lat."),
    )
    compact_names = {
        "CN_gt_Dementia": "CN–DEM",
        "MCI_gt_Dementia": "MCI–DEM",
        "CN_gt_MCI": "CN–MCI",
    }
    fig = plt.figure(figsize=(9.6, 4.9), facecolor="white")
    grid = fig.add_gridspec(
        3,
        6,
        width_ratios=(1, 1, 1, 1, 0.30, 1.25),
        left=0.015,
        right=0.99,
        bottom=0.14,
        top=0.94,
        wspace=-0.04,
        hspace=-0.22,
    )
    for row, contrast in enumerate(CONTRASTS):
        for column, (hemisphere, view, view_label) in enumerate(views):
            axis = fig.add_subplot(grid[row, column], projection="3d")
            nilearn_plotting.plot_surf_stat_map(
                fsaverage[f"infl_{hemisphere}"],
                textures[(contrast, hemisphere)],
                bg_map=fsaverage[f"sulc_{hemisphere}"],
                hemi=hemisphere,
                view=view,
                threshold=Z_THRESHOLD,
                cmap="RdBu_r",
                vmin=-Z_MAX,
                vmax=Z_MAX,
                symmetric_cbar=True,
                colorbar=False,
                bg_on_data=False,
                alpha=0.95,
                axes=axis,
                figure=fig,
            )
            for collection in axis.collections:
                collection.set_rasterized(True)
            if row == 0:
                axis.set_title(
                    view_label, fontsize=11, fontweight="semibold", pad=-3
                )
        glass_axis = fig.add_subplot(grid[row, 5])
        signed_map = np.where(
            np.abs(maps[contrast]) >= Z_THRESHOLD, maps[contrast], 0
        )
        signed_image = nib.Nifti1Image(
            signed_map.astype(np.float32),
            references[contrast].affine,
            references[contrast].header,
        )
        nilearn_plotting.plot_glass_brain(
            signed_image,
            display_mode="ortho",
            threshold=Z_THRESHOLD,
            cmap="RdBu_r",
            vmin=-Z_MAX,
            vmax=Z_MAX,
            colorbar=False,
            plot_abs=False,
            symmetric_cbar=True,
            annotate=False,
            black_bg=False,
            axes=glass_axis,
            figure=fig,
        )
        if row == 0:
            glass_axis.set_title(
                "Whole volume", fontsize=10.5, fontweight="semibold", pad=-3
            )
        row_position = grid[row, 0].get_position(fig)
        fig.text(
            0.020,
            (row_position.y0 + row_position.y1) / 2,
            compact_names[contrast],
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
        )

    common_colorbar(
        fig,
        [0.35, 0.07, 0.30, 0.018],
        label="",
        label_fontsize=12,
        tick_fontsize=10.5,
    )
    save(fig, output, "zmap_inflated_surface_all")
    plt.close(fig)


def main() -> None:
    global WORK, MNI_T1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--mni-t1", type=Path, default=MNI_T1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    WORK = args.work_dir.resolve()
    MNI_T1 = args.mni_t1.resolve()
    if args.output is None:
        args.output = WORK / "manuscript/figures"
    args.output.mkdir(parents=True, exist_ok=True)

    map_paths = {
        contrast: WORK
        / "results/application_methods/zmaps"
        / f"sigma3_{contrast}_signed_z.nii.gz"
        for contrast in CONTRASTS
    }
    references = {contrast: nib.load(str(path)) for contrast, path in map_paths.items()}
    reference = references[CONTRASTS[0]]
    background_reference = nib.load(str(MNI_T1))
    if reference.shape[:3] != background_reference.shape[:3] or not np.allclose(
        reference.affine, background_reference.affine, atol=1e-5
    ):
        raise RuntimeError("MNI background and Z maps have different geometry")
    if any(
        image.shape[:3] != reference.shape[:3]
        or not np.allclose(image.affine, reference.affine, atol=1e-5)
        for image in references.values()
    ):
        raise RuntimeError("Z maps have inconsistent geometry")

    maps = {contrast: load(path) for contrast, path in map_paths.items()}
    mask = load(WORK / "fslvbm/stats/GM_mask.nii.gz") > 0
    background = load(MNI_T1)
    combined_slice_montage(maps, references, mask, background, args.output)
    combined_mip(maps, reference, mask, args.output)
    compact_surface_figure(maps, references, args.output)


if __name__ == "__main__":
    main()
