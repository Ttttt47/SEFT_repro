from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/neurovault_790848"
SOURCE_SHA256 = "f9121c3414d2ecb5f0ec73749c6552ca19fd86bffa2b63e13400eae0b6727477"


def test_bundled_neurovault_demo_matches_provenance_and_aal3_grid() -> None:
    source_path = DEMO / "source_speechRev_Zmap_4mm.nii.gz"
    demo_path = DEMO / "speech_vs_reversed_zmap_aal3_2mm.nii.gz"
    atlas_path = ROOT / "real_data/input/atlas/AAL3v1.nii.gz"

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == SOURCE_SHA256
    source = nib.load(str(source_path))
    demo = nib.load(str(demo_path))
    atlas = nib.load(str(atlas_path))
    assert demo.shape == atlas.shape
    np.testing.assert_allclose(demo.affine, atlas.affine, rtol=0, atol=1e-6)

    atlas_data = np.asarray(atlas.dataobj)
    demo_data = np.asarray(demo.dataobj, dtype=np.float32)
    assert np.all(np.isfinite(demo_data))
    assert np.all(demo_data[atlas_data <= 0] == 0)
    assert np.any(demo_data[atlas_data > 0] > 0)
    assert np.any(demo_data[atlas_data > 0] < 0)

    expected = np.asarray(
        resample_from_to(source, atlas, order=1, mode="constant", cval=0.0).dataobj,
        dtype=np.float32,
    )
    expected = np.where((atlas_data > 0) & np.isfinite(expected), expected, 0.0)
    np.testing.assert_allclose(demo_data, expected, rtol=0, atol=1e-6)
