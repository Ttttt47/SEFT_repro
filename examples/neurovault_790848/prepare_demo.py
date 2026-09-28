#!/usr/bin/env python3
"""Resample the bundled NeuroVault Z-map to the repository AAL3 grid."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
SOURCE = HERE / "source_speechRev_Zmap_4mm.nii.gz"
ATLAS = REPRO_ROOT / "real_data/input/atlas/AAL3v1.nii.gz"
OUTPUT = HERE / "speech_vs_reversed_zmap_aal3_2mm.nii.gz"
SOURCE_SHA256 = "f9121c3414d2ecb5f0ec73749c6552ca19fd86bffa2b63e13400eae0b6727477"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    atlas_path = args.atlas.resolve()
    output = args.output.resolve()
    if source == SOURCE.resolve() and sha256(source) != SOURCE_SHA256:
        raise RuntimeError("Bundled NeuroVault source file failed its SHA-256 check")

    source_img = nib.load(str(source))
    atlas_img = nib.load(str(atlas_path))
    if len(source_img.shape) != 3 or len(atlas_img.shape) != 3:
        raise ValueError("The source Z-map and target atlas must both be three-dimensional")

    resampled = resample_from_to(source_img, atlas_img, order=1, mode="constant", cval=0.0)
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    atlas = np.asarray(atlas_img.dataobj)
    data = np.where((atlas > 0) & np.isfinite(data), data, 0.0).astype(np.float32)

    header = atlas_img.header.copy()
    header.set_data_dtype(np.float32)
    header["descrip"] = b"NeuroVault 790848 resampled to AAL3 2 mm"
    result = nib.Nifti1Image(data, atlas_img.affine, header)
    result.set_qform(atlas_img.affine, int(atlas_img.header["qform_code"]) or 2)
    result.set_sform(atlas_img.affine, int(atlas_img.header["sform_code"]) or 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))

    values = data[atlas > 0]
    print(f"Wrote {output}")
    print(f"shape={result.shape}; min={values.min():.6g}; max={values.max():.6g}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
