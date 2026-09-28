#!/usr/bin/env python3
"""Generate the main ADNI discovery-count and scientific-question figures."""

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import nibabel as nib
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[3]
WORK = REPRO_ROOT / "outputs/adni3"
RESULTS = WORK / "results/absmax_working_models_real_seed20260723/tables"
FIGURE_DIR = WORK / "manuscript/figures"
ATLAS_PATH = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
GM_MASK_PATH = WORK / "fslvbm/stats/GM_mask.nii.gz"
TEMPLATE_PATH = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "data/standard/MNI152_T1_2mm_brain.nii.gz"

PC_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
CONTRASTS = [
    ("CN_gt_Dementia", "CN > dementia"),
    ("MCI_gt_Dementia", "MCI > dementia"),
    ("CN_gt_MCI", "CN > MCI"),
]
STABILITY_DISPLAY = {
    "CN_gt_Dementia": "CN--dementia",
    "MCI_gt_Dementia": "MCI--dementia",
    "CN_gt_MCI": "CN--MCI",
}
STABILITY_PC_LEVEL = {
    "CN_gt_Dementia": 0.20,
    "MCI_gt_Dementia": 0.20,
    "CN_gt_MCI": 0.10,
}
STABILITY_CLASS_ORDER = [
    "stable", "moderately_stable", "unstable_exploratory",
]
STABILITY_CLASS_LABEL = {
    "stable": "Stable",
    "moderately_stable": "Moderately stable",
    "unstable_exploratory": "Exploratory",
}
STABILITY_CLASS_COLOR = {
    "stable": "#0072B2",
    "moderately_stable": "#E69F00",
    "unstable_exploratory": "#A6A6A6",
}
RESAMPLE_COLOR = {"bootstrap": "#6A3D9A", "delete10": "#009E73"}


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def discovery_counts() -> None:
    seft = pd.read_csv(RESULTS / "combined_region_summary_all_pc.tsv", sep="\t")
    seft = seft[(seft["method"] == "SEFT-CO") & np.isclose(seft["alpha"], 0.10)]

    bh = pd.read_csv(RESULTS / "bh_simes_region_results_all_pc.tsv", sep="\t")
    bh = (
        bh[np.isclose(bh["alpha"], 0.10)]
        .groupby(["contrast", "pc_level"], as_index=False)["significant"]
        .sum()
        .rename(columns={"significant": "discoveries"})
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.65), sharex=True, sharey=True)
    for ax, (contrast, title) in zip(axes, CONTRASTS):
        s = (
            seft[seft["contrast"] == contrast]
            .set_index("pc_level")
            .reindex(PC_LEVELS)["discoveries"]
            .to_numpy()
        )
        b = (
            bh[bh["contrast"] == contrast]
            .set_index("pc_level")
            .reindex(PC_LEVELS)["discoveries"]
            .to_numpy()
        )
        ax.plot(
            PC_LEVELS,
            s,
            color="black",
            linewidth=1.6,
            marker="+",
            markersize=8,
            markeredgewidth=1.3,
            label="SEFT",
        )
        ax.plot(
            PC_LEVELS,
            b,
            color="#00B83F",
            linewidth=1.6,
            linestyle=(0, (5, 3)),
            marker="x",
            markersize=6,
            markeredgewidth=1.2,
            label="BHe",
        )
        strip = Rectangle(
            (0, 1),
            1,
            0.145,
            transform=ax.transAxes,
            clip_on=False,
            facecolor="#f2f2f2",
            edgecolor="#3f3f3f",
            linewidth=0.8,
        )
        ax.add_patch(strip)
        ax.text(
            0.5,
            1.072,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xticks(PC_LEVELS)
        ax.set_xticklabels([".01", ".05", ".10", ".20", ".30", ".40", ".50"])
        ax.set_xlabel("PC level ($c$)")
        ax.axvline(0.20, color="#666666", linestyle=":", linewidth=1.0)
        primary_count = int(s[np.flatnonzero(np.isclose(PC_LEVELS, 0.20))[0]])
        ax.annotate(
            str(primary_count),
            xy=(0.20, primary_count),
            xytext=(5, 8 if primary_count else 10),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="black",
        )
        ax.grid(True, color="#e4e4e4", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#3f3f3f")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Number of regions")
    axes[0].set_ylim(-4, 135)
    axes[2].legend(
        frameon=True,
        edgecolor="#3f3f3f",
        fancybox=False,
        loc="upper right",
        borderpad=0.35,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.86, wspace=0.10)
    save_figure(fig, "adni_seft_bh_discovery_counts")


def resampling_stability_summary(work: Path) -> None:
    tables = work / "results/stability_primary/tables"
    regions = pd.read_csv(tables / "region_stability_summary.tsv", sep="\t")
    replicates = pd.read_csv(tables / "replicate_summary.tsv", sep="\t")
    regions = regions[regions["full_sample_selected"].eq(1)].copy()
    contrast_names = [name for name, _ in CONTRASTS]

    figure = plt.figure(figsize=(8.6, 4.75), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=[1.05, 0.78])
    scatter_axes = [
        figure.add_subplot(grid[0, 2 * index : 2 * index + 2])
        for index in range(3)
    ]

    for axis, contrast in zip(scatter_axes, contrast_names):
        block = regions[regions["contrast"].eq(contrast)]
        axis.axvspan(
            0.70, 1.0, ymin=0.70, ymax=1.0,
            color="#D9F0E3", alpha=0.55, zorder=0,
        )
        axis.axvline(
            0.70, color="#555555", linestyle=(0, (3, 3)), linewidth=0.8,
        )
        axis.axhline(
            0.70, color="#555555", linestyle=(0, (3, 3)), linewidth=0.8,
        )
        axis.plot([0, 1], [0, 1], color="#B8B8B8", linewidth=0.8, zorder=0)
        for stability_class in STABILITY_CLASS_ORDER[::-1]:
            subset = block[block["stability_class"].eq(stability_class)]
            axis.scatter(
                subset["bootstrap_frequency"],
                subset["delete10_frequency"],
                s=34,
                color=STABILITY_CLASS_COLOR[stability_class],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.92,
                zorder=3,
            )
        stable_count = int((block["stability_class"] == "stable").sum())
        axis.set_title(
            f"{STABILITY_DISPLAY[contrast]}\n"
            f"$c={STABILITY_PC_LEVEL[contrast]:.2f}$; "
            f"{stable_count}/{len(block)} stable",
            fontsize=10.5,
        )
        axis.set_xlim(-0.025, 1.025)
        axis.set_ylim(-0.025, 1.025)
        axis.set_box_aspect(1)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_xlabel("Bootstrap selection frequency", fontsize=10.5)
        axis.tick_params(labelsize=9.5)
        axis.grid(color="#ECECEC", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    scatter_axes[0].set_ylabel("Delete-10% selection frequency", fontsize=10.5)
    for axis in scatter_axes[1:]:
        axis.set_yticklabels([])

    jaccard_axis = figure.add_subplot(grid[1, :3])
    rng = np.random.default_rng(20260801)
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    for contrast_index, contrast in enumerate(contrast_names):
        for offset, resampling in zip(
            (-0.17, 0.17), ("bootstrap", "delete10")
        ):
            subset = replicates[
                replicates["contrast"].eq(contrast)
                & replicates["resampling"].eq(resampling)
            ]["jaccard_with_full_sample"].to_numpy(float)
            position = contrast_index + 1 + offset
            positions.append(position)
            values.append(subset)
            colors.append(RESAMPLE_COLOR[resampling])
            jitter = rng.normal(0, 0.028, size=len(subset))
            jaccard_axis.scatter(
                np.full(len(subset), position) + jitter,
                subset,
                s=8,
                color=RESAMPLE_COLOR[resampling],
                alpha=0.28,
                linewidth=0,
                zorder=1,
            )
    boxplot = jaccard_axis.boxplot(
        values,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.4},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.9)
    jaccard_axis.set_xticks(
        range(1, 4), [STABILITY_DISPLAY[name] for name in contrast_names]
    )
    jaccard_axis.set_ylim(-0.025, 1.025)
    jaccard_axis.set_ylabel("Jaccard with full sample", fontsize=10.5)
    jaccard_axis.tick_params(labelsize=9.5)
    jaccard_axis.grid(axis="y", color="#ECECEC", linewidth=0.6)
    jaccard_axis.set_axisbelow(True)
    jaccard_axis.spines["top"].set_visible(False)
    jaccard_axis.spines["right"].set_visible(False)
    jaccard_axis.legend(
        handles=[
            Line2D(
                [0], [0], color=RESAMPLE_COLOR["bootstrap"],
                linewidth=6, label="Bootstrap",
            ),
            Line2D(
                [0], [0], color=RESAMPLE_COLOR["delete10"],
                linewidth=6, label="Delete 10%",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        fontsize=9.5,
    )

    class_axis = figure.add_subplot(grid[1, 3:])
    count_table = (
        regions.groupby(["contrast", "stability_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=contrast_names,
            columns=STABILITY_CLASS_ORDER,
            fill_value=0,
        )
    )
    bottom = np.zeros(len(contrast_names))
    x = np.arange(len(contrast_names))
    for stability_class in STABILITY_CLASS_ORDER:
        counts = count_table[stability_class].to_numpy(float)
        class_axis.bar(
            x,
            counts,
            bottom=bottom,
            width=0.66,
            color=STABILITY_CLASS_COLOR[stability_class],
            label=STABILITY_CLASS_LABEL[stability_class],
        )
        for index, (start, count) in enumerate(zip(bottom, counts)):
            if count >= 4:
                class_axis.text(
                    index,
                    start + count / 2,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if stability_class != "moderately_stable"
                        else "#333333"
                    ),
                    fontsize=9.5,
                    fontweight="bold",
                )
        bottom += counts
    class_axis.set_xticks(
        x, [STABILITY_DISPLAY[name] for name in contrast_names]
    )
    class_axis.set_ylabel("Full-sample discoveries", fontsize=10.5)
    class_axis.tick_params(labelsize=9.5)
    class_axis.grid(axis="y", color="#ECECEC", linewidth=0.6)
    class_axis.set_axisbelow(True)
    class_axis.spines["top"].set_visible(False)
    class_axis.spines["right"].set_visible(False)
    class_axis.legend(frameon=False, ncol=1, loc="upper right", fontsize=9.5)
    figure.align_ylabels([scatter_axes[0], jaccard_axis])
    save_figure(figure, "adni_resampling_stability_summary")


def selected_regions(
    table: pd.DataFrame, contrast: str, pc_level: float
) -> set[int]:
    rows = table[
        (table["method"] == "SEFT-CO")
        & (table["contrast"] == contrast)
        & np.isclose(table["pc_level"], pc_level)
        & np.isclose(table["alpha"], 0.10)
        & (table["significant"] == 1)
    ]
    return set(rows["region_id"].astype(int))


def scale_background(background: np.ndarray) -> np.ndarray:
    positive = background[background > 0]
    lo, hi = np.percentile(positive, [1, 99.7])
    return np.clip((background - lo) / (hi - lo), 0, 1)


def overlap_montage() -> None:
    table = pd.read_csv(RESULTS / "combined_region_results_all_pc.tsv", sep="\t")
    atlas_img = nib.load(ATLAS_PATH)
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int16)
    gm = nib.load(GM_MASK_PATH).get_fdata() > 0
    background = scale_background(nib.load(TEMPLATE_PATH).get_fdata())
    affine = atlas_img.affine

    comparisons = [
        {
            "contrast": "CN_gt_MCI",
            "pc": 0.10,
        },
        {
            "contrast": "MCI_gt_Dementia",
            "pc": 0.20,
        },
    ]
    z_coords = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    shared_color = "#D81B60"
    benchmark_color = "#D9A441"
    comparison_color = "#168C95"
    overlay_cmap = ListedColormap(
        [(0, 0, 0, 0), shared_color, benchmark_color, comparison_color]
    )

    fig, axes = plt.subplots(2, len(z_coords), figsize=(10.2, 3.25))
    audit_rows = []

    for row, spec in enumerate(comparisons):
        comparison = selected_regions(table, spec["contrast"], spec["pc"])
        benchmark = selected_regions(table, "CN_gt_Dementia", spec["pc"])
        shared = comparison & benchmark
        benchmark_only = benchmark - comparison
        comparison_only = comparison - benchmark

        audit_rows.append(
            {
                "contrast": spec["contrast"],
                "pc_level": spec["pc"],
                "comparison_discoveries": len(comparison),
                "benchmark_discoveries": len(benchmark),
                "shared": len(shared),
                "benchmark_only": len(benchmark_only),
                "comparison_only": len(comparison_only),
            }
        )

        overlay = np.zeros(atlas.shape, dtype=np.uint8)
        overlay[np.isin(atlas, list(shared)) & gm] = 1
        overlay[np.isin(atlas, list(benchmark_only)) & gm] = 2
        overlay[np.isin(atlas, list(comparison_only)) & gm] = 3

        for col, z_mm in enumerate(z_coords):
            ax = axes[row, col]
            z_index = int(round((z_mm - affine[2, 3]) / affine[2, 2]))
            z_index = int(np.clip(z_index, 0, atlas.shape[2] - 1))
            actual_z = int(round(affine[2, 2] * z_index + affine[2, 3]))

            bg_slice = np.rot90(background[:, :, z_index])
            ov_slice = np.rot90(overlay[:, :, z_index])
            masked_overlay = np.ma.masked_where(ov_slice == 0, ov_slice)

            ax.imshow(bg_slice, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(
                masked_overlay,
                cmap=overlay_cmap,
                vmin=0,
                vmax=3,
                alpha=0.86,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(f"$z={actual_z}$", fontsize=11, pad=2)
            if col == 0:
                ax.text(
                    0.025,
                    0.94,
                    "R",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=9.5,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )
                ax.text(
                    0.975,
                    0.94,
                    "L",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=9.5,
                    fontweight="bold",
                    ha="right",
                    va="top",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    legend = [
        Patch(facecolor=shared_color, label="Shared"),
        Patch(facecolor=benchmark_color, label="CN--dementia only"),
        Patch(facecolor=comparison_color, label="Comparison only"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.005),
        handlelength=1.2,
        columnspacing=1.4,
        prop={"size": 10.5},
    )
    fig.subplots_adjust(
        left=0.005, right=0.997, top=0.92, bottom=0.12, wspace=0.015, hspace=0.045
    )
    save_figure(fig, "adni_scientific_question_overlap")

    pd.DataFrame(audit_rows).to_csv(
        FIGURE_DIR / "adni_scientific_question_overlap_counts.tsv",
        sep="\t",
        index=False,
    )


def main() -> None:
    global RESULTS, FIGURE_DIR, GM_MASK_PATH, TEMPLATE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mni-t1", type=Path, default=TEMPLATE_PATH)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    RESULTS = (args.results_dir or work / "results/absmax_working_models_real_seed20260723/tables").resolve()
    FIGURE_DIR = (args.output_dir or work / "manuscript/figures").resolve()
    GM_MASK_PATH = work / "fslvbm/stats/GM_mask.nii.gz"
    TEMPLATE_PATH = args.mni_t1.resolve()
    set_plot_style()
    discovery_counts()
    resampling_stability_summary(work)
    overlap_montage()


if __name__ == "__main__":
    main()
