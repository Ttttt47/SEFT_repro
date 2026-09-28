#!/usr/bin/env python3
"""Run the manuscript L=64 IID/GRF simulation with resumable outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]


def call(command: list[object], env: dict[str, str]) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], cwd=REPRO_ROOT, env=env, check=True)


def visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        return [item.strip() for item in result.stdout.splitlines() if item.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPRO_ROOT / "outputs/sim_3d")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--gpu-workers", type=int, default=0)
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", "Rscript"))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", default="common,fdrs_absmax,deepfdr_absmax,fchmrf_absmax")
    args = parser.parse_args()
    if args.workers < 1 or args.gpu_workers < 0:
        parser.error("worker counts must be nonnegative and --workers must be positive")
    gpu_ids = visible_gpu_ids()
    gpu_workers = args.gpu_workers or max(1, len(gpu_ids))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    env["SEFT_GPU_IDS"] = ",".join(gpu_ids[:gpu_workers])
    common_extra: list[object] = []
    method_extra: list[object] = []
    if args.smoke:
        smoke_geometry = ["--L", 16, "--radius", 3, "--signal-fwhm", 6,
                          "--noise-fwhm", 2, "--denoise-level", 0, "--region-size", 4]
        common_extra = ["--repetitions", 1, "--noise-types", "iid",
                        "--point-values", "2", "--task-limit", 2, *smoke_geometry]
        method_extra = ["--task-limit", 2, *smoke_geometry]
    requested = [value.strip() for value in args.methods.split(",") if value.strip()]
    if "common" in requested:
        call([args.rscript, HERE / "run_common_grid.R", "--workers", args.workers,
              "--seed", args.seed, "--output-dir", root / "common", *common_extra], env)
    for method in ("fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax"):
        if method not in requested:
            continue
        is_gpu = method == "deepfdr_absmax"
        workers = gpu_workers if is_gpu else args.workers
        call([args.rscript, HERE / "run_absmax_method.R", "--method", method,
              "--workers", workers, "--gpu-workers", gpu_workers,
              "--seed", args.seed, "--original-dir", root / "common",
              "--output-dir", root / method, *method_extra], env)
    if not args.smoke and set(requested) == {"common", "fdrs_absmax", "deepfdr_absmax", "fchmrf_absmax"}:
        call([args.rscript, HERE / "make_figures.R", root, root / "figures"], env)


if __name__ == "__main__":
    main()
