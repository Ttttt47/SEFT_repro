from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.stats import norm, t as t_dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/seft_fsl"))
from seft_fsl_lib import assert_same_geometry, design_degrees_of_freedom, tstat_to_signed_z  # noqa: E402


def test_t_to_signed_z_matches_scipy() -> None:
    values = np.asarray([-8.0, -2.5, 0.0, 2.5, 8.0])
    observed = tstat_to_signed_z(values, 307)
    expected = np.sign(values) * norm.isf(t_dist.sf(np.abs(values), 307))
    assert np.allclose(observed, expected, atol=1e-12)


def test_design_rank_and_df(tmp_path: Path) -> None:
    path = tmp_path / "design.mat"
    path.write_text("/NumWaves 2\n/NumPoints 4\n/Matrix\n1 0\n1 1\n1 2\n1 3\n")
    assert design_degrees_of_freedom(path) == (2, 4, 2)


def test_rank_deficient_design_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "design_rank_deficient.mat"
    path.write_text("/NumWaves 2\n/NumPoints 3\n/Matrix\n1 1\n1 1\n1 1\n")
    try:
        design_degrees_of_freedom(path)
    except ValueError as error:
        assert "Rank-deficient" in str(error)
    else:
        raise AssertionError("Rank-deficient FSL design was accepted")


def test_geometry_guard() -> None:
    reference = nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4))
    matching = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    assert_same_geometry(reference, matching, "reference", "matching")
    shifted = np.eye(4); shifted[0, 3] = 2
    mismatch = nib.Nifti1Image(np.ones((3, 4, 5)), shifted)
    try:
        assert_same_geometry(reference, mismatch, "reference", "mismatch")
    except ValueError:
        pass
    else:
        raise AssertionError("affine mismatch was not rejected")
