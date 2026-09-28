#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys

import numpy as np


project_root = Path(__file__).resolve().parents[1]
module_path = project_root / "python" / "working_models" / "ml_working_model.py"
sys.path.insert(0, str(module_path.parent))
specification = importlib.util.spec_from_file_location(
    "ml_working_model", module_path
)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)

rng = np.random.default_rng(20260723)
x = rng.normal(size=(7, 6, 5))
x_til = rng.normal(size=x.shape)
background = np.where(
    np.abs(x) > np.abs(x_til),
    x,
    np.where(np.abs(x_til) > np.abs(x), x_til, np.maximum(x, x_til)),
)
context = np.clip(
    1 / (1 + np.exp(-(np.abs(background) - 1))),
    1e-4,
    1 - 1e-4,
)
grid, density = module.plis_weighted_grid_density(background, context)
score_x = module.deepfdr_plis_candidate_null_scores(
    background, x, context, grid, density
)
score_til = module.deepfdr_plis_candidate_null_scores(
    background, x_til, context, grid, density
)

swap = rng.random(x.shape) < 0.3
x_swapped = x.copy()
x_til_swapped = x_til.copy()
x_swapped[swap] = x_til[swap]
x_til_swapped[swap] = x[swap]
background_swapped = np.where(
    np.abs(x_swapped) > np.abs(x_til_swapped),
    x_swapped,
    np.where(
        np.abs(x_til_swapped) > np.abs(x_swapped),
        x_til_swapped,
        np.maximum(x_swapped, x_til_swapped),
    ),
)
assert np.array_equal(background, background_swapped)

score_x_swapped = module.deepfdr_plis_candidate_null_scores(
    background_swapped, x_swapped, context, grid, density
)
score_til_swapped = module.deepfdr_plis_candidate_null_scores(
    background_swapped, x_til_swapped, context, grid, density
)
expected_x = score_x.copy()
expected_til = score_til.copy()
expected_x[swap] = score_til[swap]
expected_til[swap] = score_x[swap]
error = max(
    np.max(np.abs(score_x_swapped - expected_x)),
    np.max(np.abs(score_til_swapped - expected_til)),
)
assert error == 0.0
assert np.all(np.isfinite(score_x))
assert np.all(np.isfinite(score_til))
assert np.all((score_x > 0) & (score_x < 1))
assert np.all((score_til > 0) & (score_til < 1))
print("ML candidate reweight exchangeability passed; max error = 0.")

density_context = {
    "dataset": background.ravel(),
    "weights": context.ravel() / context.sum(),
    "bandwidth": 0.4,
}
f1_background = module.official_weighted_gaussian_kde_pdf(
    background, density_context, batch_size=32
)
fc_x = module.candidate_null_scores(
    background, x, context, density_context, f1_background
)
fc_til = module.candidate_null_scores(
    background, x_til, context, density_context, f1_background
)
fc_x_swapped = module.candidate_null_scores(
    background, x_swapped, context, density_context, f1_background
)
fc_til_swapped = module.candidate_null_scores(
    background, x_til_swapped, context, density_context, f1_background
)
expected_fc_x = fc_x.copy()
expected_fc_til = fc_til.copy()
expected_fc_x[swap] = fc_til[swap]
expected_fc_til[swap] = fc_x[swap]
fc_error = max(
    np.max(np.abs(fc_x_swapped - expected_fc_x)),
    np.max(np.abs(fc_til_swapped - expected_fc_til)),
)
assert fc_error == 0.0
print("fcHMRF candidate reweight exchangeability passed; max error = 0.")
