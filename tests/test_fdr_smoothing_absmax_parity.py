#!/usr/bin/env python3
"""Full small-lattice parity against a direct Python abs-max patch."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/working_models"))

import absmax_backend
from fdr_smoothing_absmax import load_or_create_trails


def a0(value):
    return 2.0 * stats.norm.cdf(np.abs(value)) - 1.0


def g0(value):
    return 2.0 * stats.norm.pdf(value) * a0(value)


def g1_component(value, mean):
    absolute = np.abs(value)
    return (
        stats.norm.pdf(value, loc=mean) * a0(value)
        + stats.norm.pdf(value)
        * (
            stats.norm.cdf(absolute, loc=mean)
            - stats.norm.cdf(-absolute, loc=mean)
        )
    )


class Distribution:
    def __init__(self, grid=None, density=None, null=False):
        self.grid = grid
        self.density = density
        self.null = null

    def pdf(self, values):
        if self.null:
            return g0(np.asarray(values))
        return np.interp(values, self.grid, self.density)


def edge_map(edges):
    result = defaultdict(list)
    for left, right in edges:
        result[int(left)].append(int(right))
        result[int(right)].append(int(left))
    return result


if not hasattr(np, "product"):
    np.product = np.prod

from pygfl.solver import TrailSolver
from smoothfdr.easy import solution_path_smooth_fdr

dims = (4, 4, 4)
mean_grid = np.linspace(-4.0, 4.0, 220)
table_size = 524288
table_grid = np.linspace(mean_grid[0], mean_grid[-1], table_size)
rng = np.random.default_rng(73)
indices = rng.integers(20000, table_size - 20000, size=np.prod(dims))
background = table_grid[indices]
np.random.seed(91)
order = np.concatenate(
    [np.random.permutation(background.size) for _ in range(10)]
).astype(np.int64)

cpp = absmax_backend.predictive_recursion_absmax(
    background, mean_grid, order, -0.67, table_size
)
theta = np.ones(mean_grid.size) / mean_grid.size
pi0 = 1.0
for iteration, index in enumerate(order):
    joint = g1_component(background[index], mean_grid) * theta
    m0 = pi0 * g0(background[index])
    m1 = np.trapezoid(joint, mean_grid)
    mixture = m0 + m1
    weight = (3.0 + iteration) ** -0.67
    pi0 = (1.0 - weight) * pi0 + weight * m0 / mixture
    theta = (1.0 - weight) * theta + weight * joint / mixture

alternative_mass = 1.0 - pi0
f1 = np.asarray([
    np.trapezoid(
        theta * stats.norm.pdf(value, loc=mean_grid), mean_grid
    ) / alternative_mass
    for value in mean_grid
])
g1 = np.asarray([
    np.trapezoid(
        theta * g1_component(value, mean_grid), mean_grid
    ) / alternative_mass
    for value in mean_grid
])
assert abs(float(cpp["pi0"]) - pi0) < 1e-8
assert np.max(np.abs(np.asarray(cpp["theta_subdensity"]) - theta)) < 1e-8
assert np.max(np.abs(np.log(cpp["f1"]) - np.log(f1))) < 1e-8
assert np.max(np.abs(np.log(cpp["g1"]) - np.log(g1))) < 1e-8

ntrails, trails, breakpoints, edges = load_or_create_trails(
    dims, str(ROOT / ".cache/smoothfdr_trails")
)


def solve(signal_density):
    solver = TrailSolver()
    solver.set_data(
        background,
        edge_map(edges),
        ntrails,
        trails,
        breakpoints,
    )
    return solution_path_smooth_fdr(
        background,
        solver,
        Distribution(null=True),
        Distribution(mean_grid, signal_density),
        verbose=0,
    )


fit_cpp = solve(np.asarray(cpp["g1"]))
fit_ref = solve(g1)
assert int(fit_cpp["best"]) == int(fit_ref["best"])
assert abs(float(fit_cpp["lambda"]) - float(fit_ref["lambda"])) < 1e-12
posterior_error = np.max(
    np.abs(np.asarray(fit_cpp["posteriors"]) - fit_ref["posteriors"])
)
assert posterior_error < 1e-8

candidate = rng.uniform(mean_grid[0], mean_grid[-1], background.size)
cpp_score = absmax_backend.reweight_scores(
    background,
    candidate,
    np.asarray(fit_cpp["posteriors"]),
    mean_grid,
    np.asarray(cpp["f1"]),
    np.asarray(cpp["g1"]),
)
q = np.clip(np.asarray(fit_ref["posteriors"]), 1e-12, 1.0 - 1e-12)
log_odds = (
    np.log(q)
    - np.log1p(-q)
    + np.log(np.interp(candidate, mean_grid, f1))
    - stats.norm.logpdf(candidate)
    - np.log(np.interp(background, mean_grid, g1))
    + np.log(g0(background))
)
reference_log_score = -np.logaddexp(0.0, log_odds)
score_error = np.max(
    np.abs(np.asarray(cpp_score["log_score"]) - reference_log_score)
)
assert score_error < 1e-8

print(
    "FDR_SMOOTHING_ABSMAX_PARITY_OK",
    {
        "lambda": float(fit_cpp["lambda"]),
        "posterior_max_error": float(posterior_error),
        "log_score_max_error": float(score_error),
    },
)
