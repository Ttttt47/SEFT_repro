#!/usr/bin/env python3
"""Portable entry point for regional comparisons, stability, and overlap plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGES = HERE.parent / "stages"


def run(command: list[object], env: dict[str, str]) -> None:
    subprocess.run([str(value) for value in command], check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mni-t1", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--ho-atlas", type=Path, required=True)
    parser.add_argument("--ho-labels", type=Path, required=True)
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    run([
        args.rscript, STAGES / "run_real_map_extended_checks.R", "--work-dir", args.work_dir,
        "--workers", args.workers, "--seed", args.seed,
    ], env)
    run([
        args.python, HERE / "working_models_tfce.py", "--work-dir", args.work_dir,
        "--background", args.background, "--ho-atlas", args.ho_atlas,
        "--ho-labels", args.ho_labels,
    ], env)
    run([
        args.python, HERE / "regional_results.py", "--work-dir", args.work_dir,
        "--mni-t1", args.mni_t1,
    ], env)


if __name__ == "__main__":
    main()
