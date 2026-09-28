#!/usr/bin/env python3
import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_cmd(cmd, log_file, cwd=None):
    with log_file.open("a", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(str(part) for part in cmd)}")


def find_largest_nifti(outdir, vbm_id):
    candidates = []
    for pattern in (f"{vbm_id}*.nii.gz", f"{vbm_id}*.nii"):
        candidates.extend(outdir.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def has_dicom(path):
    return path.exists() and any(path.glob("*.dcm"))


def process_sample(row, args):
    vbm_id = row["vbm_id"].strip()
    label = row["label"].strip()
    subject_id = row["subject_id"].strip()
    image_id = row["image_id"].strip()
    zip_file = row["zip_file"].strip()
    zip_path = Path(row["zip_path"].strip())

    sample_log_dir = Path(args.log_dir) / "sample_jobs"
    sample_log_dir.mkdir(parents=True, exist_ok=True)
    sample_log = sample_log_dir / f"{vbm_id}.log"
    with sample_log.open("a", encoding="utf-8") as log:
        log.write("\n--- resume/check invocation ---\n")

    raw_dir = Path(args.raw_dicom_dir) / vbm_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not has_dicom(raw_dir):
        print(f"EXTRACT {vbm_id} from {zip_file}", flush=True)
        run_cmd(["unzip", "-q", "-j", zip_path, f"*/I{image_id}/*", "-d", raw_dir], sample_log)
        if not has_dicom(raw_dir):
            raise RuntimeError(f"No DICOM files extracted for {vbm_id}")
    else:
        print(f"SKIP DICOM {vbm_id}", flush=True)

    nifti_dir = Path(args.nifti_dir) / vbm_id
    nifti_dir.mkdir(parents=True, exist_ok=True)
    src = find_largest_nifti(nifti_dir, vbm_id)
    if src is None:
        print(f"DCM2NIIX {vbm_id}", flush=True)
        run_cmd([args.dcm2niix, "-z", "y", "-f", vbm_id, "-o", nifti_dir, raw_dir], sample_log)
        src = find_largest_nifti(nifti_dir, vbm_id)
    if src is None:
        raise RuntimeError(f"No NIfTI output found for {vbm_id}")

    fslvbm_dir = Path(args.fslvbm_dir)
    fslvbm_dir.mkdir(parents=True, exist_ok=True)
    target = fslvbm_dir / f"{vbm_id}.nii.gz"
    if not target.exists() or target.stat().st_size == 0:
        print(f"REORIENT {vbm_id}", flush=True)
        run_cmd(["fslreorient2std", src, target], sample_log)

    stats = subprocess.run(
        ["fslstats", str(target), "-V", "-R", "-M"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.split()
    if len(stats) < 5:
        raise RuntimeError(f"Unexpected fslstats output for {vbm_id}: {' '.join(stats)}")

    return {
        "vbm_id": vbm_id,
        "label": label,
        "subject_id": subject_id,
        "image_id": image_id,
        "nifti_path": str(target),
        "voxels": stats[0],
        "intensity_min": stats[2],
        "intensity_max": stats[3],
        "intensity_mean": stats[4],
    }


def read_sample_plan(path):
    with Path(path).open(newline="", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description="Parallel DICOM extraction, dcm2niix conversion, and FSL-VBM input prep.")
    parser.add_argument("--sample-plan", required=True)
    parser.add_argument("--raw-dicom-dir", required=True)
    parser.add_argument("--nifti-dir", required=True)
    parser.add_argument("--fslvbm-dir", required=True)
    parser.add_argument("--qc-csv", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--dcm2niix", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    rows = read_sample_plan(args.sample_plan)
    if not rows:
        raise SystemExit("Sample plan is empty.")

    threads = min(args.threads, len(rows))
    print(f"Running sample preprocessing with {threads} parallel jobs for {len(rows)} samples", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(process_sample, row, args): row["vbm_id"] for row in rows}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["vbm_id"])
    qc_path = Path(args.qc_csv)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "vbm_id",
        "label",
        "subject_id",
        "image_id",
        "nifti_path",
        "voxels",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
    ]
    with qc_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote NIfTI QC to {qc_path}", flush=True)


if __name__ == "__main__":
    main()

