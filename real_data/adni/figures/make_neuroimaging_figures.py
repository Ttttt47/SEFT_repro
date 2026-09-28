#!/usr/bin/env python3
"""Portable entry point for ADNI statistical-map displays and diagnostics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(command: list[object]) -> None:
    subprocess.run([str(value) for value in command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mni-t1", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    run([
        args.python, HERE / "neuroimaging_diagnostics.py", "--work-dir", args.work_dir,
        "--data-dir", args.data_dir, "--mni-t1", args.mni_t1,
    ])
    run([
        args.python, HERE / "neuroimaging_views.py", "--work-dir", args.work_dir,
        "--mni-t1", args.mni_t1,
    ])


if __name__ == "__main__":
    main()
