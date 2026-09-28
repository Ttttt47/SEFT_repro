#!/usr/bin/env python3
"""Aggregate the real-data abs-max runs and make region/spatial figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import ndtr


REPRO_ROOT = Path(__file__).resolve().parents[3]
WORK = REPRO_ROOT / "outputs/adni3"
APP = WORK / "results/application_methods"
ATLAS = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
ATLAS_LABELS = ATLAS.with_name("AAL3v1.nii.txt")
HO_ATLAS = WORK / "atlas/harvard_oxford_thr25_2mm.nii.gz"
HO_LABELS = HO_ATLAS.with_name("harvard_oxford_thr25_2mm_labels.tsv")
GM_MASK = WORK / "fslvbm/stats/GM_mask.nii.gz"
BACKGROUND = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "data/standard/MNI152_T1_2mm_brain.nii.gz"
CONTRASTS = ["CN_gt_Dementia", "MCI_gt_Dementia", "CN_gt_MCI"]
DISPLAY = {
    "CN_gt_Dementia": "CN > dementia",
    "MCI_gt_Dementia": "MCI > dementia",
    "CN_gt_MCI": "CN > MCI",
}
METHOD_ORDER = [
    "SEFT-CO",
    "SEFT-Ising-AbsMaxAdapt",
    "SEFT-FDRSmoothing-AbsMaxAdapt",
    "SEFT-DeepFDR-AbsMaxAdapt",
    "SEFT-fcHMRF-AbsMaxAdapt",
]
METHOD_SHORT = {
    "SEFT-CO": "SEFT-CO",
    "SEFT-Ising-AbsMaxAdapt": "SEFT-Ising",
    "SEFT-FDRSmoothing-AbsMaxAdapt": "SEFT-FDR smoothing",
    "SEFT-DeepFDR-AbsMaxAdapt": "SEFT-DeepFDR",
    "SEFT-fcHMRF-AbsMaxAdapt": "SEFT-fcHMRF",
}
PC_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
PC_COLORS = {
    0.0: np.array([0xF1, 0xF1, 0xF1]) / 255.0,
    0.01: np.array([0xFD, 0xC5, 0x27]) / 255.0,
    0.05: np.array([0xFA, 0x8E, 0x47]) / 255.0,
    0.1: np.array([0xED, 0x5A, 0x5F]) / 255.0,
    0.2: np.array([0xC9, 0x3F, 0x73]) / 255.0,
    0.3: np.array([0x9C, 0x2E, 0x7F]) / 255.0,
    0.4: np.array([0x70, 0x1F, 0x81]) / 255.0,
    0.5: np.array([0x45, 0x10, 0x71]) / 255.0,
}
# A neutral light gray denotes no rejection.  Among rejected regions, darker
# colors encode a higher (and therefore stronger) significant PC level.
PC_LEGEND_LEVELS = PC_LEVELS

def load(path: Path, dtype=np.float32) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def aal3_labels() -> dict[int, str]:
    result: dict[int, str] = {}
    for line in ATLAS_LABELS.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            result[int(float(fields[0]))] = fields[1]
    return result


def harvard_oxford_labels() -> dict[int, str]:
    frame = pd.read_csv(HO_LABELS, sep="\t")
    return dict(zip(frame.region_id.astype(int), frame.label.astype(str)))


AAL3_ANATOMICAL_DIVISIONS = [
    (1, 2, "Frontal and sensorimotor", "Primary motor cortex"),
    (3, 12, "Frontal and sensorimotor", "Lateral frontal cortex"),
    (13, 16, "Frontal and sensorimotor", "Rolandic operculum and SMA"),
    (17, 18, "Medial and orbitofrontal", "Olfactory cortex"),
    (19, 24, "Medial and orbitofrontal", "Medial frontal and gyrus rectus"),
    (25, 32, "Medial and orbitofrontal", "Orbitofrontal cortex"),
    (33, 34, "Insula, limbic and medial temporal", "Insula"),
    (35, 40, "Insula, limbic and medial temporal", "Cingulate cortex"),
    (41, 46, "Insula, limbic and medial temporal", "Medial temporal and amygdala"),
    (47, 58, "Occipital and ventral visual", "Occipital visual cortex"),
    (59, 60, "Occipital and ventral visual", "Fusiform cortex"),
    (61, 62, "Parietal and somatosensory", "Primary somatosensory cortex"),
    (63, 74, "Parietal and somatosensory", "Parietal association cortex"),
    (75, 80, "Basal ganglia and thalamus", "Dorsal basal ganglia"),
    (81, 82, "Basal ganglia and thalamus", "Whole thalamus"),
    (83, 84, "Temporal and auditory", "Primary auditory cortex"),
    (85, 94, "Temporal and auditory", "Temporal association cortex"),
    (95, 112, "Cerebellum and vermis", "Cerebellar hemispheres"),
    (113, 120, "Cerebellum and vermis", "Cerebellar vermis"),
    (121, 150, "Thalamic nuclei", "Thalamic nuclei"),
    (151, 156, "ACC and ventral striatum", "Anterior cingulate subdivisions"),
    (157, 158, "ACC and ventral striatum", "Nucleus accumbens"),
    (159, 170, "Midbrain and brainstem nuclei", "Midbrain and brainstem nuclei"),
]

AAL3_GROUP_SHORT = {
    "Frontal and sensorimotor": "Frontal–motor",
    "Medial and orbitofrontal": "Medial/OFC",
    "Insula, limbic and medial temporal": "Insula–limbic",
    "Occipital and ventral visual": "Occipital",
    "Parietal and somatosensory": "Parietal–somatic",
    "Basal ganglia and thalamus": "BG/thalamus",
    "Temporal and auditory": "Temporal",
    "Cerebellum and vermis": "Cerebellum",
    "Thalamic nuclei": "Thalamic nuclei",
    "ACC and ventral striatum": "ACC/NAcc",
    "Midbrain and brainstem nuclei": "Midbrain",
}

COMPACT_GROUP_LABELS = {
    "Frontal–motor": "Frontal/\nmotor",
    "Medial/OFC": "Medial/\nOFC",
    "Insula–limbic": "Insular/\nlimbic",
    "Occipital": "Occipital",
    "Parietal–somatic": "Parietal/\nsom.",
    "BG/thalamus": "BG",
    "Temporal": "Temporal",
    "Cerebellum": "Cerebellum",
    "Thalamic nuclei": "Thalamic\nnuclei",
    "ACC/NAcc": "ACC/\nNAcc",
    "Midbrain": "Brainstem\nnuclei",
    "Frontal/motor": "Front./mot.",
    "Insula/opercular": "Ins./op.",
    "Limbic/MTL": "Limbic/MTL",
    "Temporal/auditory": "Temp./aud.",
    "Parietal": "Par.",
    "Occipital/visual": "Occ./vis.",
    "Subcortical": "Subcort.",
}


def aal3_division(region_id: int) -> tuple[str, str]:
    for lower, upper, macroregion, subdivision in AAL3_ANATOMICAL_DIVISIONS:
        if lower <= region_id <= upper:
            return macroregion, subdivision
    raise ValueError(f"No anatomical division for AAL3 region {region_id}.")


def harvard_oxford_group(label: str) -> str:
    value = label.lower()
    if any(token in value for token in (
        "thalamus", "caudate", "putamen", "pallidum", "accumbens",
        "brain-stem",
    )):
        return "Subcortical"
    if any(token in value for token in (
        "hippocampus", "amygdala", "parahippocampal", "cingulate",
        "paracingulate", "subcallosal",
    )):
        return "Limbic/MTL"
    if any(token in value for token in (
        "occipital", "calcarine", "cuneal", "lingual", "fusiform",
    )):
        return "Occipital/visual"
    if any(token in value for token in (
        "postcentral", "parietal", "supramarginal", "angular", "precune",
    )):
        return "Parietal"
    if any(token in value for token in (
        "temporal", "heschl", "planum",
    )):
        return "Temporal/auditory"
    if any(token in value for token in ("insula", "opercul")):
        return "Insula/opercular"
    return "Frontal/motor"


def bh(p_values: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p_values)
    ordered = p_values[order]
    passed = ordered <= alpha * np.arange(1, len(ordered) + 1) / len(ordered)
    answer = np.zeros(len(ordered), dtype=np.uint8)
    if np.any(passed):
        answer[order[: np.flatnonzero(passed)[-1] + 1]] = 1
    return answer


def gather(output: Path) -> pd.DataFrame:
    extended_path = output / "tables/combined_region_results_all_pc.tsv"
    if extended_path.exists():
        combined = pd.read_csv(extended_path, sep="\t")
        observed = sorted(combined.pc_level.unique())
        if not np.allclose(observed, PC_LEVELS):
            raise RuntimeError(
                f"All-PC table has levels {observed}, expected {PC_LEVELS}"
            )
        combined.to_csv(
            output / "tables/combined_region_results.tsv",
            sep="\t", index=False,
        )
        summary = (
            combined.groupby(
                ["method", "adaptation", "contrast", "pc_level", "alpha"],
                as_index=False,
            )
            .agg(
                discoveries=("significant", "sum"),
                n_regions=("region_id", "size"),
            )
        )
        summary.to_csv(
            output / "tables/combined_region_summary.tsv",
            sep="\t", index=False,
        )
        return combined

    frames = [
        pd.read_csv(path, sep="\t")
        for path in sorted((output / "tables").glob("*__*_regions.tsv"))
    ]
    if len(frames) != 9:
        raise RuntimeError(f"Expected 9 adapted region tables, found {len(frames)}")
    adapted = pd.concat(frames, ignore_index=True)

    co = pd.read_csv(
        APP / "tables/seft_simes_region_results_long.tsv", sep="\t"
    )
    co = co[
        co.atlas.eq("aal3")
        & co.variant.eq("sigma3")
        & co.method.eq("seft")
        & np.isclose(co.alpha, 0.1)
        & co.contrast.isin(CONTRASTS)
        & co.pc_level.isin(PC_LEVELS)
    ].copy()
    co["method"] = "SEFT-CO"
    co["adaptation"] = "manuscript_logscore"
    co["significant"] = co["significant_recomputed"].astype(int)
    keep = [
        "method", "adaptation", "contrast", "pc_level", "region_id",
        "region_label", "n_voxels", "u", "e_value", "pc_p_value",
        "significant", "alpha",
    ]
    combined = pd.concat([co[keep], adapted[keep]], ignore_index=True)
    combined["region_id"] = combined.region_id.astype(int)
    combined.to_csv(
        output / "tables/combined_region_results.tsv", sep="\t", index=False
    )
    summary = (
        combined.groupby(
            ["method", "adaptation", "contrast", "pc_level", "alpha"],
            as_index=False,
        )
        .agg(
            discoveries=("significant", "sum"),
            n_regions=("region_id", "size"),
        )
    )
    summary.to_csv(
        output / "tables/combined_region_summary.tsv", sep="\t", index=False
    )
    return combined


def tfce_table() -> pd.DataFrame:
    frame = pd.read_csv(APP / "tables/tfce_by_aal3_region.tsv", sep="\t")
    return frame[
        frame.variant.eq("sigma3") & frame.contrast.isin(CONTRASTS)
    ].copy()


def simes_bh_table(
    atlas: np.ndarray,
    gm_mask: np.ndarray,
    region_ids: list[int],
    region_labels: dict[int, str],
    output: Path,
    atlas_tag: str = "aal3",
) -> pd.DataFrame:
    """Reconstruct the existing Simes-PC/BHe reference at every plotted PC level."""
    rows: list[dict[str, object]] = []
    for contrast in CONTRASTS:
        # The retained z-model is voxelwise and independent of the regional
        # partition; its AAL3 run is the one that saved internal maps.
        prefix = f"sigma3_{contrast}_aal3"
        z_model_path = (
            APP / "seft_runs" / prefix / "maps/claw_internal"
            / f"{prefix}_z_model.nii.gz"
        )
        z_model = load(z_model_path, dtype=np.float64)
        voxel_p = 2.0 * ndtr(-np.abs(z_model))
        for pc_level in PC_LEVELS:
            level_rows: list[dict[str, object]] = []
            for region_id in region_ids:
                region_p = np.sort(voxel_p[gm_mask & (atlas == region_id)])
                n_voxels = len(region_p)
                u = max(1, int(np.ceil(pc_level * n_voxels)))
                tail = region_p[u - 1:]
                factors = (n_voxels - u + 1) / np.arange(
                    1, n_voxels - u + 2
                )
                pc_p_value = min(1.0, float(np.min(factors * tail)))
                level_rows.append({
                    "method": "BH",
                    "adaptation": "simes_pc_bh_existing_reference",
                    "contrast": contrast,
                    "pc_level": pc_level,
                    "region_id": region_id,
                    "region_label": region_labels.get(region_id, str(region_id)),
                    "n_voxels": n_voxels,
                    "u": u,
                    "e_value": np.nan,
                    "pc_p_value": pc_p_value,
                    "alpha": 0.1,
                })
            decisions = bh(
                np.array([row["pc_p_value"] for row in level_rows]), 0.1
            )
            for row, decision in zip(level_rows, decisions):
                row["significant"] = int(decision)
                rows.append(row)
    result = pd.DataFrame(rows)
    file_tag = "" if atlas_tag == "aal3" else f"_{atlas_tag}"
    result.to_csv(
        output / f"tables/bh_simes_region_results_all_pc{file_tag}.tsv",
        sep="\t", index=False,
    )

    # The released application saved PC={0.10,0.20,0.30}. Require exact
    # decision parity before using the newly reconstructed 0.01--0.50 table.
    reference = pd.read_csv(
        APP / "tables/seft_simes_region_results_long.tsv", sep="\t"
    )
    reference = reference[
        reference.atlas.eq(atlas_tag)
        & reference.variant.eq("sigma3")
        & reference.method.eq("simes")
        & np.isclose(reference.alpha, 0.1)
        & reference.contrast.isin(CONTRASTS)
    ][[
        "contrast", "pc_level", "region_id", "pc_p_value",
        "significant_recomputed",
    ]]
    audit = result[
        result.pc_level.isin([0.1, 0.2, 0.3])
    ].merge(
        reference,
        on=["contrast", "pc_level", "region_id"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    audit["p_value_abs_error"] = np.abs(
        audit.pc_p_value_new - audit.pc_p_value_reference
    )
    audit["decision_mismatch"] = (
        audit.significant != audit.significant_recomputed
    ).astype(int)
    audit.to_csv(
        output / f"tables/bh_simes_reconstruction_audit{file_tag}.tsv",
        sep="\t", index=False,
    )
    # AAL3 must reproduce the retained primary analysis exactly.  The old
    # Harvard–Oxford sensitivity run refitted its voxel model on an
    # atlas-specific support, whereas this figure deliberately holds the
    # voxel score fixed and changes only the regional partition.  Keep that
    # comparison as an audit table, but do not mix its decisions into the
    # fixed-score atlas sensitivity figure.
    if (
        atlas_tag == "aal3"
        and (
            audit.decision_mismatch.sum() != 0
            or audit.p_value_abs_error.max() > 5e-7
        )
    ):
        raise RuntimeError("Reconstructed Simes-PC/BH reference failed parity.")
    return result


def tfce_table_for_atlas(
    atlas: np.ndarray,
    gm_mask: np.ndarray,
    region_ids: list[int],
    region_labels: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for contrast_index, contrast in enumerate(CONTRASTS, start=1):
        corrp = load(
            APP / "randomise"
            / f"three_group_sigma3_tfce_corrp_tstat{contrast_index}.nii.gz"
        )
        significant = gm_mask & (corrp >= 0.95)
        for region_id in region_ids:
            region = gm_mask & (atlas == region_id)
            n_voxels = int(region.sum())
            n_significant = int((region & significant).sum())
            rows.append({
                "variant": "sigma3",
                "contrast": contrast,
                "region_id": region_id,
                "region_label": region_labels.get(region_id, str(region_id)),
                "n_voxels": n_voxels,
                "tfce_significant_voxels": n_significant,
                "tfce_coverage_fraction": (
                    n_significant / n_voxels if n_voxels else 0.0
                ),
            })
    return pd.DataFrame(rows)


def heatmap_canvas(row_rgb: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Use a narrow separator before TFCE instead of a full blank data row."""
    row_pixels = 10
    gap_pixels = 3
    method_block = np.repeat(row_rgb[:-1], row_pixels, axis=0)
    gap = np.ones((gap_pixels, row_rgb.shape[1], 3), dtype=row_rgb.dtype)
    tfce_block = np.repeat(row_rgb[-1:], row_pixels, axis=0)
    canvas = np.concatenate([method_block, gap, tfce_block], axis=0)
    centers = [
        row_pixels * index + (row_pixels - 1) / 2
        for index in range(row_rgb.shape[0] - 1)
    ]
    centers.append(
        row_pixels * (row_rgb.shape[0] - 1)
        + gap_pixels + (row_pixels - 1) / 2
    )
    return canvas, centers


def heatmap_canvas_tfce_top(
    row_rgb: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Place TFCE alone above a gap, followed by one continuous method block."""
    row_pixels = 10
    gap_pixels = 3
    tfce_block = np.repeat(row_rgb[:1], row_pixels, axis=0)
    gap = np.ones((gap_pixels, row_rgb.shape[1], 3), dtype=row_rgb.dtype)
    method_block = np.repeat(row_rgb[1:], row_pixels, axis=0)
    canvas = np.concatenate([tfce_block, gap, method_block], axis=0)
    centers = [(row_pixels - 1) / 2]
    centers.extend([
        row_pixels + gap_pixels
        + row_pixels * index + (row_pixels - 1) / 2
        for index in range(row_rgb.shape[0] - 1)
    ])
    return canvas, centers


def compact_highlight_regions(
    combined: pd.DataFrame,
    contrast: str,
    region_ids: list[int],
    region_labels: dict[int, str],
    pair_hemispheres: bool,
    pc_threshold: float,
    min_methods: int,
    max_labels: int = 8,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Select a few result-driven labels for the compact heatmap.

    For AAL3, adjacent left/right regions are combined into a single
    anatomical callout. Ranking first favors bilateral robustness and then
    the number of supporting methods.
    """
    selected = combined[
        combined.contrast.eq(contrast) & combined.significant.eq(1)
    ]
    maximum = (
        selected.groupby(["method", "region_id"]).pc_level.max()
        .unstack("method")
        .reindex(index=region_ids, columns=METHOD_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    position = {region_id: index for index, region_id in enumerate(region_ids)}
    rows = []
    for region_id in region_ids:
        label = region_labels.get(region_id, str(region_id))
        values = maximum.loc[region_id].to_numpy(dtype=float)
        if pair_hemispheres:
            base = label.removesuffix("_L").removesuffix("_R")
        else:
            base = label
        side = (
            "L" if label.endswith("_L")
            else "R" if label.endswith("_R")
            else ""
        )
        rows.append({
            "region_id": region_id,
            "region_label": label,
            "base_label": base,
            "side": side,
            "position": position[region_id],
            "strong_methods": int(
                np.sum(values >= pc_threshold - 1e-12)
            ),
            "mean_max_pc": float(np.mean(values)),
        })
    region_frame = pd.DataFrame(rows)
    grouped_rows = []
    for base_label, group in region_frame.groupby("base_label", sort=False):
        strong = group[group.strong_methods.ge(min_methods)]
        if strong.empty:
            continue
        strong_sides = set(strong.side) - {""}
        if strong_sides == {"L", "R"}:
            side_text = "L/R"
        elif len(strong_sides) == 1:
            side_text = next(iter(strong_sides))
        else:
            side_text = ""
        clean_label = base_label.replace("_", " ")
        annotation = (
            f"{clean_label} ({side_text})" if side_text else clean_label
        )
        grouped_rows.append({
            "contrast": contrast,
            "annotation": annotation,
            "base_label": base_label,
            "x_position": float(strong.position.mean()),
            "region_ids": ",".join(str(value) for value in group.region_id),
            "region_labels": ";".join(group.region_label),
            "n_sides_meeting_rule": int(len(strong)),
            "strong_method_count_sum": int(group.strong_methods.sum()),
            "mean_max_pc": float(group.mean_max_pc.mean()),
            "pc_threshold": pc_threshold,
            "minimum_supporting_methods": min_methods,
            "selection_rule": (
                f"At least {min_methods} of 5 SEFT scores significant "
                f"at PC={pc_threshold:.2f}"
            ),
        })
    highlight_frame = pd.DataFrame(grouped_rows)
    if highlight_frame.empty:
        return [], highlight_frame
    highlight_frame = (
        highlight_frame.sort_values(
            [
                "n_sides_meeting_rule",
                "strong_method_count_sum",
                "mean_max_pc",
                "x_position",
            ],
            ascending=[False, False, False, True],
        )
        .head(max_labels)
        .sort_values("x_position")
        .reset_index(drop=True)
    )
    return highlight_frame.to_dict("records"), highlight_frame


def style_heatmap_axis(
    axis: plt.Axes,
    canvas: np.ndarray,
    row_centers: list[float],
    row_labels: list[str],
    region_ids: list[int],
    region_labels: dict[int, str],
    y_fontsize: float,
    x_fontsize: float,
    major_groups: dict[int, str] | None = None,
    group_fontsize: float = 16.0,
    group_separator_width: float = 5.0,
    compact_annotations: list[dict[str, object]] | None = None,
    group_label_stagger: float = 0.0,
    group_separator_color: str = "#30343B",
    group_label_overrides: dict[str, str] | None = None,
    group_label_box: bool = True,
    group_label_position: str = "top",
    frame_color: str | None = None,
    frame_width: float = 1.0,
) -> None:
    axis.imshow(canvas, interpolation="nearest", aspect="auto")
    axis.set_yticks(row_centers)
    axis.set_yticklabels(row_labels, fontsize=y_fontsize)
    if compact_annotations is not None:
        for tick_label, row_label in zip(
            axis.get_yticklabels(), row_labels
        ):
            if row_label == "SEFT-CO":
                tick_label.set_fontweight("bold")
    if compact_annotations is None:
        axis.set_xticks(np.arange(len(region_ids)))
        axis.set_xticklabels(
            [region_labels.get(value, str(value)) for value in region_ids],
            rotation=90, fontsize=x_fontsize,
        )
    else:
        axis.set_xticks([])
        occupied_intervals: list[list[tuple[float, float]]] = []
        annotation_bottom = canvas.shape[0] - 0.5
        annotation_target_y = row_centers[-1]
        for value in compact_annotations:
            x_position = float(value["x_position"])
            label = str(value["annotation"])
            text_x = float(np.clip(
                x_position + 2.0,
                1.0, len(region_ids) - 2.0,
            ))
            estimated_width = max(6.0, 0.85 * len(label))
            interval = (
                text_x - estimated_width / 2,
                text_x + estimated_width / 2,
            )
            lane = 0
            while lane < len(occupied_intervals):
                if all(
                    interval[1] + 2.5 < other[0]
                    or interval[0] - 2.5 > other[1]
                    for other in occupied_intervals[lane]
                ):
                    break
                lane += 1
            if lane == len(occupied_intervals):
                occupied_intervals.append([])
            occupied_intervals[lane].append(interval)
            label_y = (
                annotation_bottom
                + canvas.shape[0] * (0.095 + 0.085 * lane)
            )
            axis.annotate(
                label,
                xy=(x_position, annotation_target_y),
                xytext=(text_x, label_y),
                xycoords="data", textcoords="data",
                ha="center", va="top",
                fontsize=x_fontsize, fontweight="bold",
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#20242A",
                    "linewidth": 1.4,
                    "mutation_scale": 8,
                    "shrinkA": 2.0,
                    "shrinkB": 2.0,
                },
            )
        axis.set_xlim(-0.5, len(region_ids) - 0.5)
    # Draw only internal cell boundaries. The old -0.5 and n-0.5 grid
    # positions appeared as two unwanted vertical frame lines.
    axis.set_xticks(np.arange(0.5, len(region_ids) - 0.5, 1), minor=True)
    # Keep the cell gridlines extremely subtle: visible at native resolution,
    # but effectively disappearing when the figure is reduced for layout.
    axis.grid(which="minor", axis="x", color="white", linewidth=0.15)
    if major_groups:
        ordered_groups = [major_groups[value] for value in region_ids]
        starts = [0] + [
            index for index in range(1, len(ordered_groups))
            if ordered_groups[index] != ordered_groups[index - 1]
        ]
        ends = [value - 1 for value in starts[1:]] + [
            len(ordered_groups) - 1
        ]
        for group_index, (start, end) in enumerate(zip(starts, ends)):
            if group_index:
                axis.axvline(
                    start - 0.5, color=group_separator_color,
                    linewidth=group_separator_width, zorder=5,
                )
            group_label = ordered_groups[start]
            if group_label_overrides:
                group_label = group_label_overrides.get(
                    group_label, group_label
                )
            label_below = group_label_position == "bottom"
            axis.text(
                (start + end) / 2,
                (
                    -0.055 - group_label_stagger * (group_index % 2)
                    if label_below
                    else 1.016 + group_label_stagger * (group_index % 2)
                ),
                group_label,
                transform=axis.get_xaxis_transform(),
                ha="center", va="top" if label_below else "bottom",
                fontsize=group_fontsize,
                fontweight="bold", clip_on=False,
                bbox=({
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#E6E9ED",
                    "edgecolor": "#69717A",
                    "linewidth": 0.8,
                    "alpha": 1.0,
                } if group_label_box else None),
            )
    for boundary in row_centers[:-1]:
        axis.axhline(boundary + 5, color="white", linewidth=0.25)
    for spine in axis.spines.values():
        spine.set_visible(frame_color is not None)
        if frame_color is not None:
            spine.set_color(frame_color)
            spine.set_linewidth(frame_width)
            spine.set_zorder(6)
    axis.tick_params(which="both", length=0)


def region_heatmap(
    combined: pd.DataFrame,
    bh_results: pd.DataFrame,
    tfce: pd.DataFrame,
    region_ids: list[int],
    region_labels: dict[int, str],
    figures: Path,
    atlas_tag: str,
    atlas_display: str,
    major_groups: dict[int, str],
) -> None:
    compact_atlas = len(region_ids) <= 70
    figure_width = 46.0 if compact_atlas else 28.0
    fig, axes = plt.subplots(
        len(CONTRASTS), 1,
        figsize=(figure_width, 24.0 if compact_atlas else 18.0),
        constrained_layout=True,
    )
    compact_panels: list[dict[str, object]] = []
    compact_highlight_tables: list[pd.DataFrame] = []
    # Match the PC encoding direction: darker means greater TFCE coverage.
    tfce_cmap = plt.get_cmap("Greys")
    for axis, contrast in zip(axes, CONTRASTS):
        # A non-rejected PC hypothesis is neutral gray. BHe is the final
        # region-level row, immediately above a narrowly separated TFCE row.
        heatmap_methods = METHOD_ORDER + ["BH"]
        rgb = np.empty((len(heatmap_methods) + 1, len(region_ids), 3))
        rgb[:] = PC_COLORS[0.0]
        subset = pd.concat(
            [
                combined[combined.contrast.eq(contrast)],
                bh_results[bh_results.contrast.eq(contrast)],
            ],
            ignore_index=True,
        )
        for method_index, method in enumerate(heatmap_methods):
            method_frame = subset[subset.method.eq(method)]
            selected = method_frame[method_frame.significant.eq(1)]
            maximum_level = selected.groupby("region_id").pc_level.max()
            for region_index, region_id in enumerate(region_ids):
                if region_id in maximum_level.index:
                    observed_level = float(maximum_level.loc[region_id])
                    level = min(
                        PC_LEVELS, key=lambda value: abs(value - observed_level)
                    )
                    rgb[method_index, region_index] = PC_COLORS[level]
        tfce_frame = tfce[tfce.contrast.eq(contrast)].set_index("region_id")
        for region_index, region_id in enumerate(region_ids):
            coverage = (
                float(tfce_frame.loc[region_id, "tfce_coverage_fraction"])
                if region_id in tfce_frame.index else 0.0
            )
            rgb[-1, region_index] = tfce_cmap(
                np.clip(coverage, 0, 1)
            )[:3]
        canvas, heatmap_row_positions = heatmap_canvas(rgb)
        heatmap_row_labels = (
            [METHOD_SHORT[value] for value in METHOD_ORDER]
            + ["BHe", "TFCE coverage"]
        )
        highlight_pc = 0.10 if contrast == "CN_gt_MCI" else 0.50
        highlight_min_methods = 1 if contrast == "CN_gt_MCI" else 3
        highlights, highlight_table = compact_highlight_regions(
            combined, contrast, region_ids, region_labels,
            pair_hemispheres=(atlas_tag == "aal3"),
            pc_threshold=highlight_pc,
            min_methods=highlight_min_methods,
        )
        # Compact layout swaps the original top CO row with the bottom TFCE
        # row.  CO is isolated below the narrow separator, so every callout
        # can terminate directly in the corresponding SEFT-CO cell.
        compact_order = [
            len(heatmap_methods),  # TFCE
            1, 2, 3, 4, 5, 0,    # other methods, BH, then CO
        ]
        compact_rgb = rgb[compact_order]
        compact_canvas, compact_row_positions = heatmap_canvas_tfce_top(
            compact_rgb
        )
        compact_row_labels = [
            "TFCE",
            "SEFT-Ising",
            "SEFT-FDR smoothing",
            "SEFT-DeepFDR",
            "SEFT-fcHMRF",
            "BHe",
            "SEFT-CO",
        ]
        compact_panels.append({
            "contrast": contrast,
            "canvas": compact_canvas,
            "row_positions": compact_row_positions,
            "row_labels": compact_row_labels,
            "highlights": highlights,
        })
        if not highlight_table.empty:
            compact_highlight_tables.append(highlight_table)
        x_fontsize = 12.0 if compact_atlas else 11.5
        group_fontsize = 17.0 if compact_atlas else 16.0
        style_heatmap_axis(
            axis, canvas, heatmap_row_positions, heatmap_row_labels,
            region_ids, region_labels, y_fontsize=17,
            x_fontsize=x_fontsize, major_groups=major_groups,
            group_fontsize=group_fontsize,
            group_label_overrides={
                "Parietal–somatic": "Parietal",
                "BG/thalamus": "BG/thal.",
            },
            group_separator_width=5.0,
        )
        axis.set_title(
            DISPLAY[contrast], loc="left", fontsize=24, pad=49,
            fontweight="bold",
        )

        # Also emit one full-width panel per contrast.  The combined figure is
        # convenient for comparison, while these versions keep 166 AAL3 labels
        # legible in a manuscript or on screen.
        single_fig, single_axis = plt.subplots(
            figsize=(
                figure_width,
                9.5 if compact_atlas else 8.5,
            )
        )
        style_heatmap_axis(
            single_axis, canvas, heatmap_row_positions, heatmap_row_labels,
            region_ids, region_labels, y_fontsize=18,
            x_fontsize=12.5 if compact_atlas else 12.0,
            major_groups=major_groups,
            group_fontsize=18.0 if compact_atlas else 17.0,
            group_separator_width=5.5,
        )
        single_axis.set_title(
            f"{atlas_display} — {DISPLAY[contrast]}: "
            "highest significant PC level by region "
            "(BHo α=0.10; higher levels overwrite lower levels)",
            loc="left", fontsize=25, pad=52, fontweight="bold",
        )
        single_fig.legend(
            handles=[Patch(
                facecolor=PC_COLORS[0.0], edgecolor="#BDBDBD",
                label="Not selected",
            )] + [
                Patch(facecolor=PC_COLORS[level], label=f"PC={level:.2f}")
                for level in PC_LEGEND_LEVELS
            ],
            loc="upper center", ncol=len(PC_LEGEND_LEVELS) + 1,
            frameon=False,
            bbox_to_anchor=(0.5, 1.17), fontsize=17,
            columnspacing=1.8, handlelength=1.6,
        )
        single_colorbar = single_fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=tfce_cmap),
            ax=single_axis, orientation="horizontal",
            fraction=0.04, pad=0.12, aspect=55,
        )
        single_colorbar.set_label(
            f"TFCE-significant voxel fraction within {atlas_display} region "
            "(voxelwise FWER 0.05)", fontsize=17,
        )
        single_colorbar.ax.tick_params(labelsize=13)
        file_tag = "" if atlas_tag == "aal3" else f"_{atlas_tag}"
        for suffix in ("pdf",):
            single_fig.savefig(
                figures / (
                    f"real_region_pc_tfce_heatmap{file_tag}_"
                    f"{contrast}.{suffix}"
                ),
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight", facecolor="white",
            )
        plt.close(single_fig)
    legend = [Patch(
        facecolor=PC_COLORS[0.0], edgecolor="#BDBDBD",
        label="Not selected",
    )] + [
        Patch(facecolor=PC_COLORS[level], label=f"PC={level:.2f}")
        for level in PC_LEGEND_LEVELS
    ]
    fig.legend(
        handles=legend, loc="upper center",
        ncol=len(PC_LEGEND_LEVELS) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01), fontsize=17,
        columnspacing=1.8, handlelength=1.6,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=tfce_cmap),
        ax=axes, orientation="horizontal", fraction=0.018, pad=0.015,
        aspect=45,
    )
    colorbar.set_label(
        f"TFCE-significant voxel fraction within {atlas_display} region",
        fontsize=17,
    )
    colorbar.ax.tick_params(labelsize=13)
    file_tag = "" if atlas_tag == "aal3" else f"_{atlas_tag}"
    for suffix in ("pdf",):
        fig.savefig(
            figures / f"real_region_pc_tfce_heatmap{file_tag}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
        if atlas_tag == "aal3":
            fig.savefig(
                figures / f"adni_regional_working_models_tfce_full166.{suffix}",
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight", facecolor="white",
            )
    plt.close(fig)

    # A manuscript-friendly alternative: omit the dense atlas tick labels and
    # retain only a few objectively selected, very strong result callouts.
    compact_width = 19.0 if compact_atlas else 23.0
    compact_fig, compact_axes = plt.subplots(
        len(CONTRASTS), 1,
        figsize=(compact_width, 15.0),
        constrained_layout=False,
    )
    compact_fig.subplots_adjust(
        left=0.12, right=0.79, top=0.96, bottom=0.235,
        hspace=0.48,
    )
    for axis, panel in zip(compact_axes, compact_panels):
        highlights = panel["highlights"]
        style_heatmap_axis(
            axis,
            panel["canvas"],
            panel["row_positions"],
            panel["row_labels"],
            region_ids,
            region_labels,
            y_fontsize=18,
            x_fontsize=14,
            major_groups=major_groups,
            group_fontsize=14.0,
            group_separator_width=1.2,
            compact_annotations=[],
            group_label_stagger=0.0,
            group_separator_color="#686868",
            group_label_overrides=COMPACT_GROUP_LABELS,
            group_label_box=False,
            group_label_position="bottom",
            frame_color="#000000",
            frame_width=1.8,
        )
        contrast = str(panel["contrast"])
        axis.set_title(
            DISPLAY[contrast], loc="left", fontsize=23, pad=12,
            fontweight="bold",
        )
    compact_legend = [
        Patch(
            facecolor=PC_COLORS[0.0], edgecolor="#BDBDBD",
            label="Not sig.",
        )
    ] + [
        Patch(facecolor=PC_COLORS[level], label=f"{level:.2f}")
        for level in PC_LEGEND_LEVELS
    ]
    compact_pc_legend = compact_fig.legend(
        handles=compact_legend, loc="center left",
        ncol=1, frameon=False,
        bbox_to_anchor=(0.795, 0.52), fontsize=18,
        labelspacing=0.45, handlelength=1.15,
        title="PC level", title_fontsize=20,
    )
    compact_pc_legend._legend_box.align = "left"
    compact_pc_legend.get_title().set_ha("left")
    compact_colorbar_axis = compact_fig.add_axes(
        [0.28, 0.167, 0.35, 0.018]
    )
    compact_colorbar = compact_fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=tfce_cmap),
        cax=compact_colorbar_axis, orientation="horizontal",
    )
    compact_colorbar.ax.tick_params(labelsize=17)
    # Place the two legend blocks side by side. Their captions share one
    # baseline and remain directly below their corresponding keys.
    compact_fig.text(
        0.455, 0.128,
        "TFCE-significant voxel fraction",
        ha="center", va="center", fontsize=18,
    )
    compact_stem = (
        "real_region_pc_tfce_heatmap"
        f"{file_tag}_compact"
    )
    for suffix in ("pdf",):
        compact_fig.savefig(
            figures / f"{compact_stem}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
        if atlas_tag == "aal3":
            compact_fig.savefig(
                figures / f"adni_regional_working_models_tfce.{suffix}",
                dpi=400 if suffix == "png" else None,
                bbox_inches="tight", facecolor="white",
            )
    plt.close(compact_fig)

    highlight_output = (
        pd.concat(compact_highlight_tables, ignore_index=True)
        if compact_highlight_tables
        else pd.DataFrame(columns=[
            "contrast", "annotation", "base_label", "x_position",
            "region_ids", "region_labels",
            "n_sides_meeting_rule", "strong_method_count_sum",
            "mean_max_pc", "pc_threshold", "minimum_supporting_methods",
            "selection_rule",
        ])
    )
    highlight_output.to_csv(
        figures.parent / "tables" / (
            f"compact_highlight_regions{file_tag}.tsv"
        ),
        sep="\t", index=False,
    )


def slice_plane(array: np.ndarray, orientation: str, index: int) -> np.ndarray:
    if orientation == "Axial":
        plane = array[:, :, index]
    elif orientation == "Coronal":
        plane = array[:, index, :]
    else:
        plane = array[index, :, :]
    return np.rot90(plane)


def make_montage(
    base: np.ndarray,
    zmap: np.ndarray,
    selected: np.ndarray,
    orientation: str,
    indices: list[int],
) -> tuple[np.ndarray, np.ma.MaskedArray]:
    bases: list[np.ndarray] = []
    overlays: list[np.ma.MaskedArray] = []
    for index in indices:
        base_plane = slice_plane(base, orientation, index)
        z_plane = slice_plane(zmap, orientation, index)
        mask_plane = slice_plane(selected, orientation, index)
        bases.append(base_plane)
        overlays.append(np.ma.masked_where(~mask_plane, z_plane))
    separator = np.zeros((bases[0].shape[0], 2), dtype=np.float32)
    masked_separator = np.ma.masked_array(
        np.zeros(separator.shape, dtype=np.float32),
        mask=np.ones(separator.shape, dtype=bool),
    )
    base_montage = bases[0]
    overlay_montage = overlays[0]
    for base_plane, overlay_plane in zip(bases[1:], overlays[1:]):
        base_montage = np.concatenate(
            [base_montage, separator, base_plane], axis=1
        )
        overlay_montage = np.ma.concatenate(
            [overlay_montage, masked_separator, overlay_plane], axis=1
        )
    return base_montage, overlay_montage


def spatial_figures(
    combined: pd.DataFrame,
    atlas: np.ndarray,
    gm_mask: np.ndarray,
    figures: Path,
    maps: Path,
) -> None:
    base = load(BACKGROUND)
    base = np.clip(base / np.quantile(base[base > 0], 0.995), 0, 1)
    coordinates = np.argwhere(gm_mask)
    index_sets = {
        "Axial": np.quantile(coordinates[:, 2], [0.2, 0.4, 0.6, 0.8]).astype(int).tolist(),
        "Coronal": np.quantile(coordinates[:, 1], [0.2, 0.4, 0.6, 0.8]).astype(int).tolist(),
        "Sagittal": np.quantile(coordinates[:, 0], [0.2, 0.4, 0.6, 0.8]).astype(int).tolist(),
    }
    for contrast_index, contrast in enumerate(CONTRASTS, start=1):
        z_path = APP / "zmaps" / f"sigma3_{contrast}_signed_z.nii.gz"
        zmap = load(z_path)
        # Signed-Z maps may carry nonfinite sentinels outside the analysis
        # mask.  They are never displayed, but replacing them avoids masked
        # image-normalisation warnings in Matplotlib/PDF backends.
        zmap = np.nan_to_num(zmap, nan=0.0, posinf=0.0, neginf=0.0)
        corrp = load(
            APP / "randomise"
            / f"three_group_sigma3_tfce_corrp_tstat{contrast_index}.nii.gz"
        )
        selections: dict[str, np.ndarray] = {}
        for method in METHOD_ORDER:
            frame = combined[
                combined.contrast.eq(contrast)
                & combined.method.eq(method)
                & np.isclose(combined.pc_level, 0.2)
                & combined.significant.eq(1)
            ]
            selected_ids = frame.region_id.astype(int).to_numpy()
            selections[METHOD_SHORT[method]] = gm_mask & np.isin(
                atlas, selected_ids
            )
        selections["TFCE"] = gm_mask & (corrp >= 0.95)

        for name, selection in selections.items():
            safe_name = name.lower().replace(" ", "_")
            output_map = maps / f"{contrast}__{safe_name}__display_mask.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    selection.astype(np.uint8),
                    nib.load(str(z_path)).affine,
                    nib.load(str(z_path)).header,
                ),
                str(output_map),
            )

        names = list(selections)
        fig, axes = plt.subplots(
            3, len(names), figsize=(22, 4.65),
            gridspec_kw={"wspace": 0.02, "hspace": 0.03},
        )
        vmax = max(3.0, float(np.quantile(np.abs(zmap[gm_mask]), 0.995)))
        last_overlay = None
        for row, orientation in enumerate(("Axial", "Coronal", "Sagittal")):
            for column, name in enumerate(names):
                axis = axes[row, column]
                base_montage, overlay_montage = make_montage(
                    base, zmap, selections[name], orientation,
                    index_sets[orientation],
                )
                axis.imshow(base_montage, cmap="gray", vmin=0, vmax=1)
                if overlay_montage.count() > 0:
                    last_overlay = axis.imshow(
                        overlay_montage, cmap="coolwarm",
                        vmin=-vmax, vmax=vmax, interpolation="nearest",
                    )
                axis.set_facecolor("black")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(name, fontsize=12)
                if column == 0:
                    axis.set_ylabel(
                        orientation, fontsize=12, color="black", labelpad=8
                    )
        colorbar_source = last_overlay if last_overlay is not None else (
            ScalarMappable(norm=Normalize(-vmax, vmax), cmap="coolwarm")
        )
        colorbar = fig.colorbar(
            colorbar_source, ax=axes, orientation="horizontal",
            fraction=0.025, pad=0.025, aspect=60,
        )
        colorbar.set_label(
            "Signed Z in selected AAL3 regions "
            "(TFCE column: FWER-significant voxels)"
        )
        fig.suptitle(
            f"{DISPLAY[contrast]}: real-data spatial distribution "
            "(PC=0.20, BHo α=0.10)",
            fontsize=15, y=0.99,
        )
        for suffix in ("pdf",):
            fig.savefig(
                figures / f"real_spatial_distribution_{contrast}.{suffix}",
                dpi=220 if suffix == "png" else None,
                bbox_inches="tight", facecolor="white",
            )
        plt.close(fig)


def main() -> None:
    global WORK, APP, GM_MASK, BACKGROUND, HO_ATLAS, HO_LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--background", type=Path, default=BACKGROUND)
    parser.add_argument("--ho-atlas", type=Path, default=HO_ATLAS)
    parser.add_argument("--ho-labels", type=Path, default=HO_LABELS)
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
    )
    args = parser.parse_args()
    WORK = args.work_dir.resolve()
    APP = WORK / "results/application_methods"
    GM_MASK = WORK / "fslvbm/stats/GM_mask.nii.gz"
    BACKGROUND = args.background.resolve()
    HO_ATLAS = args.ho_atlas.resolve()
    HO_LABELS = args.ho_labels.resolve()
    output = (args.output_dir or WORK / "results/absmax_working_models_real_seed20260723").resolve()
    figures = output / "figures"
    maps = output / "maps"
    figures.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)

    gm_mask = load(GM_MASK) > 0

    atlas = np.rint(load(ATLAS)).astype(int)
    lookup = aal3_labels()
    region_ids = sorted(
        int(value) for value in np.unique(atlas[gm_mask]) if value > 0
    )
    aal3_groups = {
        region_id: AAL3_GROUP_SHORT[aal3_division(region_id)[0]]
        for region_id in region_ids
    }
    display_order = {
        region_id: index + 1 for index, region_id in enumerate(region_ids)
    }
    aal3_mapping = pd.DataFrame([
        {
            "atlas_id_order": region_id,
            "display_order": display_order.get(region_id, np.nan),
            "included_in_gm_mask": int(region_id in display_order),
            "region_id": region_id,
            "region_label": lookup[region_id],
            "major_group": aal3_division(region_id)[0],
            "major_group_short": AAL3_GROUP_SHORT[
                aal3_division(region_id)[0]
            ],
            "anatomical_subdivision": aal3_division(region_id)[1],
        }
        for region_id in sorted(lookup)
    ])
    aal3_mapping.to_csv(
        output / "tables/aal3_display_anatomical_groups.tsv",
        sep="\t", index=False,
    )
    (
        aal3_mapping.groupby(
            ["major_group", "major_group_short", "anatomical_subdivision"],
            sort=False, as_index=False,
        )
        .agg(
            first_region_id=("region_id", "min"),
            last_region_id=("region_id", "max"),
            n_atlas_regions=("region_id", "size"),
            n_regions_in_gm_mask=("included_in_gm_mask", "sum"),
        )
        .to_csv(
            output / "tables/aal3_anatomical_group_summary.tsv",
            sep="\t", index=False,
        )
    )
    combined = gather(output)
    bh_results = simes_bh_table(
        atlas, gm_mask, region_ids, lookup, output, atlas_tag="aal3"
    )
    tfce = tfce_table()
    region_heatmap(
        combined, bh_results, tfce, region_ids, lookup, figures,
        atlas_tag="aal3", atlas_display="AAL3",
        major_groups=aal3_groups,
    )
    spatial_figures(combined, atlas, gm_mask, figures, maps)

    ho_atlas = np.rint(load(HO_ATLAS)).astype(int)
    ho_lookup = harvard_oxford_labels()
    ho_region_ids = sorted(
        int(value) for value in np.unique(ho_atlas[gm_mask]) if value > 0
    )
    ho_group_order = [
        "Frontal/motor", "Insula/opercular", "Limbic/MTL",
        "Temporal/auditory", "Parietal", "Occipital/visual",
        "Subcortical",
    ]
    ho_groups = {
        region_id: harvard_oxford_group(ho_lookup[region_id])
        for region_id in ho_region_ids
    }
    ho_region_ids.sort(
        key=lambda region_id: (
            ho_group_order.index(ho_groups[region_id]), region_id
        )
    )
    pd.DataFrame([
        {
            "display_order": index + 1,
            "region_id": region_id,
            "region_label": ho_lookup[region_id],
            "major_group": ho_groups[region_id],
        }
        for index, region_id in enumerate(ho_region_ids)
    ]).to_csv(
        output / "tables/harvard_oxford_display_anatomical_groups.tsv",
        sep="\t", index=False,
    )
    ho_combined_path = (
        output / "tables"
        / "combined_region_results_all_pc_harvard_oxford.tsv"
    )
    if not ho_combined_path.exists():
        raise RuntimeError(
            "Missing Harvard–Oxford all-PC table; run "
            "recompute_adni_all_pc_from_scores.R OUTPUT harvard_oxford."
        )
    ho_combined = pd.read_csv(ho_combined_path, sep="\t")
    if not np.allclose(sorted(ho_combined.pc_level.unique()), PC_LEVELS):
        raise RuntimeError("Harvard–Oxford table has an unexpected PC grid.")
    ho_bh = simes_bh_table(
        ho_atlas, gm_mask, ho_region_ids, ho_lookup, output,
        atlas_tag="harvard_oxford",
    )
    ho_tfce = tfce_table_for_atlas(
        ho_atlas, gm_mask, ho_region_ids, ho_lookup
    )
    ho_tfce.to_csv(
        output / "tables/tfce_by_harvard_oxford_region.tsv",
        sep="\t", index=False,
    )
    region_heatmap(
        ho_combined, ho_bh, ho_tfce, ho_region_ids, ho_lookup, figures,
        atlas_tag="harvard_oxford",
        atlas_display="Harvard–Oxford (25%, 2 mm)",
        major_groups=ho_groups,
    )
    (output / "PLOTS_COMPLETE").write_text("COMPLETE\n")


if __name__ == "__main__":
    main()
