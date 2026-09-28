#!/usr/bin/env python3
"""Compare exact and scalable max-aware KDE inside the fcHMRF adapter."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/working_models"))

from absmax_models import reweight_candidate
from ml_working_model import fit_fchmrf_context, set_deterministic


def arguments(mode):
    return argparse.Namespace(
        em_steps=5,
        learning_rate=1e-4,
        threshold=0.05,
        vendor_root=os.environ.get("SEFT_ML_VENDOR_ROOT", str(ROOT / ".vendor")),
        kde_mode=mode,
        kde_grid_size=4096,
        kde_max_grid_size=65536,
        kde_tolerance=1e-3,
    )


rng = np.random.default_rng(812)
dims = (12, 12, 12)
x = rng.normal(size=dims)
x_til = rng.normal(size=dims)
background = np.where(
    np.abs(x) > np.abs(x_til),
    x,
    np.where(np.abs(x_til) > np.abs(x), x_til, np.maximum(x, x_til)),
)

set_deterministic(1)
q_fast, kde_fast, _ = fit_fchmrf_context(
    background, background, arguments("fast"), absmax=True
)
fast_x, _ = reweight_candidate(background, x, q_fast, kde_fast)
fast_til, _ = reweight_candidate(
    background, x_til, q_fast, kde_fast
)

set_deterministic(1)
q_exact, kde_exact, _ = fit_fchmrf_context(
    background, background, arguments("exact"), absmax=True
)
exact_x, _ = reweight_candidate(background, x, q_exact, kde_exact)
exact_til, _ = reweight_candidate(
    background, x_til, q_exact, kde_exact
)

posterior_mae = float(np.mean(np.abs(q_fast - q_exact)))
score_mae = max(
    float(np.mean(np.abs(fast_x - exact_x))),
    float(np.mean(np.abs(fast_til - exact_til))),
)
rank_correlation = min(
    float(np.corrcoef(rankdata(fast_x.ravel()), rankdata(exact_x.ravel()))[0, 1]),
    float(
        np.corrcoef(
            rankdata(fast_til.ravel()), rankdata(exact_til.ravel())
        )[0, 1]
    ),
)
assert posterior_mae <= 1e-3, posterior_mae
assert score_mae <= 1e-3, score_mae
assert rank_correlation >= 0.999, rank_correlation

print(
    "FCHMRF_ABSMAX_KDE_PARITY_OK",
    {
        "posterior_mae": posterior_mae,
        "score_mae": score_mae,
        "rank_correlation": rank_correlation,
    },
)
