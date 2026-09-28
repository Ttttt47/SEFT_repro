#!/usr/bin/env python3
"""Shared helpers for the SEFT/FSL command line wrappers."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.stats import beta, norm, t as t_dist


DEFAULT_PC_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_ALPHA = 0.1
DEFAULT_BANDWIDTH = 5.0
DEFAULT_LAMBDA = 0.5
DEFAULT_NEIGHBOR_RANGE = 10
WORKING_MODEL_ALIASES = {
    "co": "co",
    "covariate-adaptive": "co",
    "fdr-smoothing": "fdr-smoothing",
    "fdrs": "fdr-smoothing",
    "deepfdr": "deepfdr",
    "fchmrf": "fchmrf",
    "ising": "ising",
}


@dataclass(frozen=True)
class SeftRunResult:
    prefix: str
    out_dir: Path
    signed_z: Path
    region_results: Path
    significant_regions: Path
    summary: Path
    metadata: Path


def parse_float_list(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("empty comma-separated float list")
        return [float(piece) for piece in pieces]
    return [float(item) for item in value]


def format_pc_level(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def read_design_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    rows: list[list[float]] = []
    in_matrix = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "/Matrix":
            in_matrix = True
            continue
        if not in_matrix:
            continue
        rows.append([float(piece) for piece in line.split()])
    if not rows:
        raise ValueError(f"No /Matrix rows found in {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Ragged design matrix in {path}")
    return np.asarray(rows, dtype=float)


def read_vest_contrast_count(path: str | Path) -> int | None:
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("/NumContrasts"):
            return int(line.split()[1])
    return None


def design_degrees_of_freedom(path: str | Path) -> tuple[int, int, int]:
    matrix = read_design_matrix(path)
    rank = int(np.linalg.matrix_rank(matrix))
    n_points = int(matrix.shape[0])
    if rank != matrix.shape[1]:
        raise ValueError(
            f"Rank-deficient design matrix: rank {rank}, {matrix.shape[1]} columns"
        )
    if n_points <= rank:
        raise ValueError(
            f"Design has no residual degrees of freedom: {n_points} rows, rank {rank}"
        )
    return n_points - rank, n_points, rank


def write_design_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"/NumWaves {matrix.shape[1]}\n")
        f.write(f"/NumPoints {matrix.shape[0]}\n")
        f.write("/PPheights " + " ".join(["1"] * matrix.shape[1]) + "\n")
        f.write("/Matrix\n")
        for row in matrix:
            f.write(" ".join(f"{value:.10g}" for value in row) + "\n")


def shuffled_design(
    design_matrix: np.ndarray,
    group_columns_1_based: Iterable[int],
    rng: np.random.Generator,
) -> np.ndarray:
    columns = [int(col) - 1 for col in group_columns_1_based]
    if not columns:
        raise ValueError("At least one group column is required")
    if min(columns) < 0 or max(columns) >= design_matrix.shape[1]:
        raise ValueError("Shuffle group column is outside the design matrix")
    permuted = np.array(design_matrix, copy=True)
    order = rng.permutation(design_matrix.shape[0])
    permuted[:, columns] = design_matrix[order][:, columns]
    return permuted


def tstat_to_signed_z(t_data: np.ndarray, df: int) -> np.ndarray:
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive; got {df}")
    t_data = np.asarray(t_data, dtype=np.float64)
    finite = np.isfinite(t_data)
    p_values = np.ones_like(t_data, dtype=np.float64)
    p_values[finite] = 2.0 * t_dist.sf(np.abs(t_data[finite]), df)
    p_values = np.clip(p_values, np.finfo(float).tiny, 1.0)
    z_abs = norm.isf(p_values / 2.0)
    z = np.sign(t_data) * z_abs
    z[~finite] = 0.0
    z[t_data == 0] = 0.0
    return z.astype(np.float32)


def geometry_summary(img: nib.Nifti1Image) -> dict[str, object]:
    return {
        "shape": list(img.shape[:3]),
        "zooms": [float(v) for v in img.header.get_zooms()[:3]],
        "affine": np.asarray(img.affine).round(6).tolist(),
    }


def assert_same_geometry(
    reference: nib.Nifti1Image,
    candidate: nib.Nifti1Image,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference.shape[:3] != candidate.shape[:3]:
        raise ValueError(
            f"Geometry mismatch: {candidate_name} shape {candidate.shape[:3]} "
            f"!= {reference_name} shape {reference.shape[:3]}"
        )
    if not np.allclose(reference.affine, candidate.affine, atol=1e-5):
        diff = float(np.max(np.abs(reference.affine - candidate.affine)))
        raise ValueError(
            f"Geometry mismatch: {candidate_name} affine differs from "
            f"{reference_name}; max abs diff {diff:g}"
        )


def save_like(data: np.ndarray, ref_img: nib.Nifti1Image, path: str | Path, dtype=np.float32) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref_img.affine, header)
    nib.save(out, str(path))
    return path


def load_region_labels(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".xml":
        import xml.etree.ElementTree as ET

        labels: dict[int, str] = {}
        root = ET.parse(path).getroot()
        for label in root.findall(".//label"):
            index = label.attrib.get("index")
            if index is None:
                continue
            labels[int(index) + 1] = (label.text or "").strip()
        return labels

    labels = {}
    with path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else None
        if "region_id" in sample.splitlines()[0].lower():
            reader = csv.DictReader(f, delimiter=delimiter or ",")
            for row in reader:
                labels[int(float(row["region_id"]))] = row.get("label", "")
            return labels
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pieces = line.split()
            if len(pieces) < 2:
                continue
            try:
                region_id = int(float(pieces[0]))
            except ValueError:
                continue
            labels[region_id] = pieces[1]
    return labels


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def build_significance_maps(
    atlas_img: nib.Nifti1Image,
    region_results_path: str | Path,
    maps_dir: str | Path,
    prefix: str,
) -> list[Path]:
    maps_dir = Path(maps_dir)
    atlas = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    rows = read_tsv(region_results_path)
    outputs: list[Path] = []
    keys = sorted({(row["pc_level"], row["method"]) for row in rows})
    for pc_level, method in keys:
        sig_ids = {
            int(float(row["region_id"]))
            for row in rows
            if row["pc_level"] == pc_level
            and row["method"] == method
            and int(float(row["significant"])) == 1
        }
        level_name = format_pc_level(float(pc_level))
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", method.lower()).strip("_")
        region_map = np.where(np.isin(atlas, list(sig_ids)), atlas, 0).astype(np.int16)
        mask_map = (region_map != 0).astype(np.uint8)
        outputs.append(
            save_like(
                region_map,
                atlas_img,
                maps_dir / f"{prefix}_pc{level_name}_{slug}_sig_regions.nii.gz",
                dtype=np.int16,
            )
        )
        outputs.append(
            save_like(
                mask_map,
                atlas_img,
                maps_dir / f"{prefix}_pc{level_name}_{slug}_sig_mask.nii.gz",
                dtype=np.uint8,
            )
        )
    return outputs


def find_repo_root(start: str | Path | None = None) -> Path:
    start_path = Path(start or __file__).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "R/core/CLAW_functions.R").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the SEFT_repro root")


def default_rscript() -> str:
    return os.environ.get("RSCRIPT", "Rscript")


def run_command(cmd: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_seft_core_r(
    *,
    signed_z: Path,
    atlas: Path,
    mask: Path | None,
    atlas_labels: Path | None,
    out_dir: Path,
    prefix: str,
    working_model: str,
    alpha: float,
    pc_levels: list[float],
    simes: bool,
    seed: int,
    denoise: str,
    bandwidth: float,
    lambda_value: float,
    neighbor_range: int,
    score_clip_c: float = 0.99,
    save_internals: bool = False,
    rscript: str | None = None,
) -> None:
    script = Path(__file__).resolve().with_name("seft_fsl_core.R")
    seft_root = find_repo_root()
    cmd = [
        rscript or default_rscript(),
        str(script),
        "--zmap",
        str(signed_z),
        "--atlas",
        str(atlas),
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
        "--working-model",
        working_model,
        "--alpha",
        str(alpha),
        "--pc-levels",
        ",".join(f"{level:g}" for level in pc_levels),
        "--seed",
        str(seed),
        "--denoise",
        denoise,
        "--bandwidth",
        str(bandwidth),
        "--lambda",
        str(lambda_value),
        "--neighbor-range",
        str(neighbor_range),
        "--score-clip-c",
        str(score_clip_c),
        "--seft-repro-dir",
        str(seft_root),
    ]
    if mask is not None:
        cmd.extend(["--mask", str(mask)])
    if atlas_labels is not None:
        cmd.extend(["--atlas-labels", str(atlas_labels)])
    if simes:
        cmd.append("--simes")
    if save_internals:
        cmd.append("--save-internals")
    run_command(cmd)


def run_seft_analysis(
    *,
    zmap: str | Path | None,
    tstat: str | Path | None,
    design_mat: str | Path | None,
    atlas: str | Path,
    atlas_labels: str | Path | None,
    mask: str | Path | None,
    out_dir: str | Path,
    prefix: str,
    working_model: str = "co",
    alpha: float,
    pc_levels: list[float],
    simes: bool,
    seed: int,
    denoise: str,
    bandwidth: float,
    lambda_value: float,
    neighbor_range: int,
    score_clip_c: float = 0.99,
    save_internals: bool = False,
    rscript: str | None = None,
    build_maps: bool = True,
) -> SeftRunResult:
    if (zmap is None) == (tstat is None):
        raise ValueError("Provide exactly one input mode: --zmap or --tstat")
    if tstat is not None and design_mat is None:
        raise ValueError("--tstat requires --design-mat")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if not pc_levels or any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in pc_levels):
        raise ValueError("Every PC level must be finite and in (0, 1]")
    if denoise not in {"none", "wavelet"}:
        raise ValueError("denoise must be 'none' or 'wavelet'")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    if not np.isfinite(lambda_value) or not (0.0 < lambda_value < 1.0):
        raise ValueError("lambda must be finite and in (0, 1)")
    if neighbor_range < 1:
        raise ValueError("neighbor range must be a positive integer")
    if not np.isfinite(score_clip_c) or not (0.0 < score_clip_c < 1.0):
        raise ValueError("score clip must be finite and in (0, 1)")
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be one portable filename component")
    if working_model not in WORKING_MODEL_ALIASES:
        raise ValueError(f"working model must be one of: {', '.join(WORKING_MODEL_ALIASES)}")
    working_model = WORKING_MODEL_ALIASES[working_model]

    out_dir = Path(out_dir)
    maps_dir = out_dir / "maps"
    tables_dir = out_dir / "tables"
    maps_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    stat_path = Path(tstat or zmap)  # type: ignore[arg-type]
    stat_img = nib.load(str(stat_path))
    if len(stat_img.shape) != 3:
        raise ValueError(f"Statistic map must be 3-D; got shape {stat_img.shape}")
    atlas_path = Path(atlas)
    atlas_img = nib.load(str(atlas_path))
    if len(atlas_img.shape) != 3:
        raise ValueError(f"Atlas must be 3-D; got shape {atlas_img.shape}")
    assert_same_geometry(stat_img, atlas_img, "statistic map", "atlas")
    atlas_data = np.asarray(atlas_img.dataobj)
    if not np.all(np.isfinite(atlas_data)) or not np.allclose(atlas_data, np.rint(atlas_data), atol=1e-5):
        raise ValueError("Atlas must contain finite integer-valued region labels")

    mask_path = Path(mask) if mask else None
    if mask_path is not None:
        mask_img = nib.load(str(mask_path))
        if len(mask_img.shape) != 3:
            raise ValueError(f"Mask must be 3-D; got shape {mask_img.shape}")
        assert_same_geometry(stat_img, mask_img, "statistic map", "mask")
        mask_data = mask_img.get_fdata() > 0
    else:
        mask_data = None

    df = None
    n_points = None
    rank = None
    if tstat is not None:
        df, n_points, rank = design_degrees_of_freedom(design_mat)  # type: ignore[arg-type]
        signed_z = tstat_to_signed_z(stat_img.get_fdata(), df)
    else:
        signed_z = np.nan_to_num(stat_img.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mask_data is not None:
        signed_z = np.where(mask_data, signed_z, 0.0).astype(np.float32)

    signed_z_path = save_like(signed_z, stat_img, maps_dir / f"{prefix}_signed_z.nii.gz", dtype=np.float32)
    run_seft_core_r(
        signed_z=signed_z_path,
        atlas=atlas_path,
        mask=mask_path,
        atlas_labels=Path(atlas_labels) if atlas_labels else None,
        out_dir=tables_dir,
        prefix=prefix,
        working_model=working_model,
        alpha=alpha,
        pc_levels=pc_levels,
        simes=simes,
        seed=seed,
        denoise=denoise,
        bandwidth=bandwidth,
        lambda_value=lambda_value,
        neighbor_range=neighbor_range,
        score_clip_c=score_clip_c,
        save_internals=save_internals,
        rscript=rscript,
    )

    region_results = tables_dir / f"{prefix}_region_results.tsv"
    significant_regions = tables_dir / f"{prefix}_significant_regions.tsv"
    summary = tables_dir / f"{prefix}_summary.tsv"
    map_outputs = build_significance_maps(atlas_img, region_results, maps_dir, prefix) if build_maps else []

    metadata = {
        "prefix": prefix,
        "working_model": working_model,
        "input_mode": "tstat" if tstat is not None else "zmap",
        "statistic_map": str(stat_path),
        "design_mat": str(design_mat) if design_mat else None,
        "df": df,
        "n_points": n_points,
        "design_rank": rank,
        "atlas": str(atlas_path),
        "atlas_labels": str(atlas_labels) if atlas_labels else None,
        "mask": str(mask_path) if mask_path else None,
        "alpha": alpha,
        "pc_levels": pc_levels,
        "simes": simes,
        "seed": seed,
        "denoise": denoise,
        "bandwidth": bandwidth,
        "lambda": lambda_value,
        "neighbor_range": neighbor_range,
        "score_clip_c": score_clip_c,
        "save_internals": save_internals,
        "geometry": {
            "statistic_map": geometry_summary(stat_img),
            "atlas": geometry_summary(atlas_img),
        },
        "outputs": {
            "signed_z": str(signed_z_path),
            "region_results": str(region_results),
            "significant_regions": str(significant_regions),
            "summary": str(summary),
            "working_model_fit": str(tables_dir / f"{prefix}_working_model_fit.rds"),
            "maps": [str(path) for path in map_outputs],
            "working_model_internal_dir": str(
                maps_dir / ("claw_internal" if working_model == "co" else f"{working_model.replace('-', '_')}_internal")
            ) if save_internals else None,
        },
    }
    metadata_path = out_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return SeftRunResult(
        prefix=prefix,
        out_dir=out_dir,
        signed_z=signed_z_path,
        region_results=region_results,
        significant_regions=significant_regions,
        summary=summary,
        metadata=metadata_path,
    )


def binomial_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return lower, upper


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)
