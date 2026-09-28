#!/usr/bin/env python3
"""Restartable orchestrator for the manuscript ADNI3 reproduction workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("index", "cohort", "convert", "vbm", "primary", "working-models", "stability", "figures", "validate")
REPRO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent / "stages"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def command_text(command: list[object]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def run(command: list[object], log_path: Path, *, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {command_text(command)}\n")
        handle.flush()
        result = subprocess.run([str(item) for item in command], cwd=REPRO_ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"[exit] {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"Stage command failed ({result.returncode}): {command_text(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=REPRO_ROOT / "outputs/adni3")
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated ordered stage subset")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--gpu-workers", type=int, default=0,
                        help="DeepFDR workers; 0 selects one per visible GPU, or one CPU fallback")
    parser.add_argument("--fsldir", type=Path, default=Path(os.environ.get("FSLDIR", "/usr/local/fsl")))
    parser.add_argument("--dcm2niix", default="dcm2niix")
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT"),
                        help="Rscript executable; defaults to SEFT_ENV/bin/Rscript")
    parser.add_argument(
        "--seft-env", type=Path,
        default=Path(os.environ.get("SEFT_ENV", Path(sys.executable).resolve().parent.parent)),
        help="Environment prefix containing the R/Python dependencies used by SEFT",
    )
    parser.add_argument("--ml-python", default=os.environ.get("SEFT_ML_PYTHON", sys.executable))
    parser.add_argument("--vendor-root", type=Path, default=Path(os.environ.get("SEFT_ML_VENDOR_ROOT", REPRO_ROOT / ".vendor")))
    parser.add_argument("--ho-atlas", type=Path)
    parser.add_argument("--ho-labels", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    requested = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = set(requested).difference(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.n_perm < 1:
        parser.error("--n-perm must be positive")
    if (args.ho_atlas is None) != (args.ho_labels is None):
        parser.error("--ho-atlas and --ho-labels must be supplied together")

    data = args.data_dir.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs/pipeline"
    rscript = args.rscript or str(args.seft_env.resolve() / "bin/Rscript")
    env = os.environ.copy()
    env.update({
        "FSLDIR": str(args.fsldir.resolve()),
        "PATH": f"{args.fsldir.resolve()}/share/fsl/bin:{args.fsldir.resolve()}/bin:" + env.get("PATH", ""),
        "SEFT_ML_PYTHON": str(args.ml_python),
        "SEFT_ML_VENDOR_ROOT": str(args.vendor_root.resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    ho_atlas = (args.ho_atlas or work / "atlas/harvard_oxford_thr25_2mm.nii.gz").resolve()
    ho_labels = (args.ho_labels or work / "atlas/harvard_oxford_thr25_2mm_labels.tsv").resolve()
    py = sys.executable
    prepare_ho = [] if args.ho_atlas and args.ho_labels else [[
        py, REPRO_ROOT / "tools/seft_fsl/seft_prepare_fsl_atlas",
        "--fsldir", args.fsldir, "--out-dir", work / "atlas",
    ]]

    commands: dict[str, list[list[object]]] = {
        "index": [[py, STAGE_DIR / "index_archives.py", "--data-dir", data, "--workers", args.workers]],
        "cohort": [[py, STAGE_DIR / "prepare_cohort.py", "--data-dir", data,
                    "--work-dir", work]],
        "convert": [[py, STAGE_DIR / "convert_images.py", "--sample-plan", work / "tables/sample_plan.tsv",
                     "--raw-dicom-dir", work / "raw_dicom", "--nifti-dir", work / "nifti",
                     "--fslvbm-dir", work / "fslvbm", "--qc-csv", work / "tables/conversion_qc.csv",
                     "--log-dir", work / "logs/conversion", "--dcm2niix", args.dcm2niix,
                     "--threads", args.workers]],
        "vbm": [[py, STAGE_DIR / "run_vbm_preprocessing.py", "--work-dir", work,
                 "--fsldir", args.fsldir, "--workers", args.workers]],
        "primary": prepare_ho + [[py, STAGE_DIR / "run_primary_methods.py", "--work-dir", work,
                     "--fsldir", args.fsldir, "--seft-env", args.seft_env,
                     "--ho-atlas", ho_atlas, "--ho-labels", ho_labels, "--threads", args.workers,
                     "--seft-jobs", args.workers, "--n-perm", args.n_perm, "--seed", args.seed]],
        "working-models": [["bash", STAGE_DIR / "run_working_models.sh", "--work-dir", work,
                            "--workers", args.workers, "--gpu-workers", args.gpu_workers,
                            "--rscript", rscript, "--python", args.ml_python]],
        "stability": [[py, STAGE_DIR / "run_stability_analysis.py", "--work-dir", work,
                       "--seft-env", args.seft_env, "--jobs", args.workers, "--seed", args.seed]],
        "figures": [
            [py, FIGURE_DIR / "make_neuroimaging_figures.py", "--work-dir", work,
             "--data-dir", data, "--mni-t1", args.fsldir / "data/standard/MNI152_T1_2mm_brain.nii.gz"],
            [py, FIGURE_DIR / "make_regional_figures.py", "--work-dir", work,
             "--mni-t1", args.fsldir / "data/standard/MNI152_T1_2mm_brain.nii.gz",
             "--background", args.fsldir / "data/standard/MNI152_T1_2mm_brain.nii.gz",
             "--ho-atlas", ho_atlas, "--ho-labels", ho_labels, "--rscript", rscript,
             "--workers", args.workers, "--seed", args.seed],
        ],
        "validate": [[py, STAGE_DIR / "validate_results.py", "--work-dir", work]],
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "repro_root": str(REPRO_ROOT),
        "data_dir": str(data), "work_dir": str(work), "stages": requested,
        "workers": args.workers, "gpu_workers": args.gpu_workers, "seed": args.seed,
        "n_perm_requested": args.n_perm, "fsldir": str(args.fsldir.resolve()),
        "seft_env": str(args.seft_env.resolve()), "rscript": rscript,
    }
    provenance = work / "provenance/pipeline_invocation.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for stage in requested:
        for command in commands[stage]:
            run(command, logs / f"{stage}.log", env=env)
    print(f"Completed stages: {', '.join(requested)}")


if __name__ == "__main__":
    main()
