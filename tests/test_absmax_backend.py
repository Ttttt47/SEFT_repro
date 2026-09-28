#!/usr/bin/env python3
"""Numerical reference checks for the shared abs-max C++ backend."""

from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/working_models"))

import absmax_backend
from absmax_models import fit_weighted_kde, reweight_candidate


def a0(value):
    return 2.0 * stats.norm.cdf(np.abs(value)) - 1.0


def g0(value):
    return 2.0 * stats.norm.pdf(value) * a0(value)


def g1_component(value, mean, sd):
    return (
        stats.norm.pdf(value, mean, sd) * a0(value)
        + stats.norm.pdf(value)
        * (
            stats.norm.cdf(np.abs(value), mean, sd)
            - stats.norm.cdf(-np.abs(value), mean, sd)
        )
    )


grid = np.linspace(-9.0, 9.0, 300001)
primitive = absmax_backend.density_primitives(grid)
assert np.max(np.abs(np.exp(primitive["log_g0"]) - g0(grid))) < 2e-14
assert abs(np.trapezoid(np.exp(primitive["log_g0"]), grid) - 1.0) < 2e-8
for mean in (-2.5, 0.0, 1.75):
    observed = np.exp(
        absmax_backend.signal_component_logpdf(grid, mean, 1.0, True)
    )
    expected = g1_component(grid, mean, 1.0)
    assert np.max(np.abs(observed - expected)) < 3e-14
    assert abs(np.trapezoid(observed, grid) - 1.0) < 2e-8


rng = np.random.default_rng(44)
dataset = rng.normal(size=80)
weights = rng.uniform(size=80)
weights /= weights.sum()
evaluation = np.linspace(-4, 4, 121)
bandwidth = 0.37
exact = absmax_backend.exact_weighted_kde(
    dataset, weights, evaluation, bandwidth
)
reference_f1 = np.sum(
    weights[None, :]
    * stats.norm.pdf(
        evaluation[:, None], dataset[None, :], bandwidth
    ),
    axis=1,
)
reference_g1 = np.sum(
    weights[None, :]
    * g1_component(
        evaluation[:, None], dataset[None, :], bandwidth
    ),
    axis=1,
)
assert np.max(np.abs(exact["f1"] - reference_f1)) < 2e-14
assert np.max(np.abs(exact["g1"] - reference_g1)) < 2e-14


context = fit_weighted_kde(
    dataset,
    weights,
    mode="fast",
    initial_grid_size=4096,
    maximum_grid_size=65536,
    tolerance=1e-3,
)
assert context.max_log_error_f1 <= 1e-3
assert context.max_log_error_g1 <= 1e-3


# Exact reference for predictive recursion. Values are table-grid points, so
# this isolates the C++ algorithm from the production interpolation.
means = np.linspace(-4, 4, 41)
values = means[np.asarray([5, 11, 19, 31, 37, 7, 18, 29])]
order = np.asarray([0, 3, 5, 2, 1, 7, 6, 4] * 2, dtype=np.int64)
cpp_pr = absmax_backend.predictive_recursion_absmax(
    values, means, order, -0.67, means.size
)
theta = np.ones(means.size) / means.size
pi0 = 1.0
for iteration, index in enumerate(order):
    joint = g1_component(values[index], means, 1.0) * theta
    m0 = pi0 * g0(values[index])
    m1 = np.trapezoid(joint, means)
    mixture = m0 + m1
    weight = (3.0 + iteration) ** -0.67
    pi0 = (1 - weight) * pi0 + weight * m0 / mixture
    theta = (1 - weight) * theta + weight * joint / mixture
assert abs(cpp_pr["pi0"] - pi0) < 1e-12
assert np.max(np.abs(cpp_pr["theta_subdensity"] - theta)) < 1e-12


# The common local reweighting must be exactly equivariant under arbitrary
# pair swaps because every fitted quantity is a function of the shared B.
x = rng.normal(size=(4, 5, 6))
x_til = rng.normal(size=x.shape)
background = np.where(
    np.abs(x) > np.abs(x_til),
    x,
    np.where(np.abs(x_til) > np.abs(x), x_til, np.maximum(x, x_til)),
)
signal = np.clip(rng.uniform(size=x.shape), 0.01, 0.99)
pair_context = fit_weighted_kde(
    background, signal, mode="exact"
)
r, log_r = reweight_candidate(background, x, signal, pair_context)
r_til, log_r_til = reweight_candidate(
    background, x_til, signal, pair_context
)
swap = rng.uniform(size=x.shape) < 0.35
x_swapped = np.where(swap, x_til, x)
x_til_swapped = np.where(swap, x, x_til)
r_swapped, log_r_swapped = reweight_candidate(
    background, x_swapped, signal, pair_context
)
r_til_swapped, log_r_til_swapped = reweight_candidate(
    background, x_til_swapped, signal, pair_context
)
assert np.max(np.abs(r_swapped - np.where(swap, r_til, r))) < 1e-14
assert np.max(np.abs(r_til_swapped - np.where(swap, r, r_til))) < 1e-14
assert np.max(
    np.abs(log_r_swapped - np.where(swap, log_r_til, log_r))
) < 1e-14
assert np.max(
    np.abs(log_r_til_swapped - np.where(swap, log_r, log_r_til))
) < 1e-14

print("ABSMAX_BACKEND_REFERENCE_OK")
