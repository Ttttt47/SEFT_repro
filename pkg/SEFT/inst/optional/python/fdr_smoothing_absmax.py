#!/usr/bin/env python3
"""Abs-max PLIS adaptation of Tansey et al.'s FDR smoothing.

The graph, greedy trail decomposition, GFL solver, lambda path, EM limits, and
BIC rule are the released ``smoothfdr`` defaults.  Only the background
emissions and predictive-recursion kernel are replaced by their abs-max
counterparts.  Numerical density work is delegated to the local C++ backend.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import absmax_backend
try:
    import persistent_gfl_backend
except ImportError:
    persistent_gfl_backend = None


class AbsMaxNullDistribution:
    def __init__(self):
        self._cached_input = None
        self._cached_pdf = None

    def pdf(self, values):
        if values is self._cached_input:
            return self._cached_pdf
        flat = np.ascontiguousarray(values, dtype=np.float64).ravel()
        result = np.exp(absmax_backend.density_primitives(flat)["log_g0"])
        result = np.asarray(result).reshape(np.shape(values))
        self._cached_input = values
        self._cached_pdf = result
        return result


class InterpolatedDistribution:
    def __init__(self, grid, density):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.density = np.maximum(
            np.asarray(density, dtype=np.float64), 1e-300
        )
        self._cached_input = None
        self._cached_pdf = None

    def pdf(self, values):
        if values is self._cached_input:
            return self._cached_pdf
        result = np.interp(
            np.asarray(values, dtype=np.float64),
            self.grid,
            self.density,
        )
        self._cached_input = values
        self._cached_pdf = result
        return result


class PersistentTrailSolver:
    """TrailSolver-compatible released GFL with reusable work arrays."""

    def __init__(
        self,
        alpha=2.0,
        inflate=2.0,
        maxsteps=100000,
        converge=1e-6,
        gfl_threads=1,
    ):
        if persistent_gfl_backend is None:
            raise ImportError(
                "persistent_gfl_backend is not built; run "
                "scripts/build_persistent_gfl_backend.sh"
            )
        self.alpha = alpha
        self.inflate = inflate
        self.maxsteps = maxsteps
        self.converge = converge
        self.gfl_threads = int(gfl_threads)
        if self.gfl_threads <= 0:
            raise ValueError("gfl_threads must be positive")

    def set_data(
        self, y, edges, ntrails, trails, breakpoints, weights=None
    ):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.edges = edges
        self.nnodes = len(y)
        self.ntrails = ntrails
        self.trails = np.ascontiguousarray(trails, dtype=np.int32)
        self.breakpoints = np.ascontiguousarray(
            breakpoints, dtype=np.int32
        )
        self.weights = weights
        self.beta = np.zeros(self.nnodes, dtype=np.float64)
        self.z = np.zeros(int(self.breakpoints[-1]), dtype=np.float64)
        self.u = np.zeros(int(self.breakpoints[-1]), dtype=np.float64)
        self.steps = []
        self.workspace = persistent_gfl_backend.PersistentGFLWorkspace(
            self.nnodes, self.trails, self.breakpoints, self.gfl_threads
        )

    def set_values_only(self, y, weights=None):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.weights = (
            None
            if weights is None
            else np.ascontiguousarray(weights, dtype=np.float64)
        )

    def solve(self, lam):
        if hasattr(lam, "__len__"):
            raise ValueError("Persistent solver accepts one lambda at a time")
        weights = (
            np.ones(self.nnodes, dtype=np.float64)
            if self.weights is None
            else self.weights
        )
        steps = self.workspace.solve(
            self.y,
            weights,
            self.beta,
            self.z,
            self.u,
            float(lam),
            self.alpha,
            self.inflate,
            self.maxsteps,
            self.converge,
        )
        self.steps.append(steps)
        return self.beta


class GridPrimalDualSolver:
    """Direct C++ solver for the same weighted 3-D graph-TV subproblem."""

    def __init__(
        self,
        dims,
        maxsteps=400,
        converge=1e-5,
        gfl_threads=1,
    ):
        if persistent_gfl_backend is None:
            raise ImportError(
                "persistent_gfl_backend is not built; run "
                "scripts/build_persistent_gfl_backend.sh"
            )
        self.dims = tuple(int(value) for value in dims)
        self.maxsteps = int(maxsteps)
        self.converge = float(converge)
        self.gfl_threads = int(gfl_threads)
        self.steps = []

    def set_data(
        self, y, edges, ntrails, trails, breakpoints, weights=None
    ):
        del ntrails, trails, breakpoints
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.edges = edges
        self.nnodes = len(y)
        self.weights = weights
        self.beta = np.zeros(self.nnodes, dtype=np.float64)
        self.z = np.zeros(3 * self.nnodes, dtype=np.float64)
        self.u = np.zeros(0, dtype=np.float64)
        self.workspace = persistent_gfl_backend.GridTVWorkspace(
            np.ascontiguousarray(self.dims, dtype=np.int32),
            self.gfl_threads,
        )

    def set_values_only(self, y, weights=None):
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.weights = (
            None
            if weights is None
            else np.ascontiguousarray(weights, dtype=np.float64)
        )

    def solve(self, lam):
        if hasattr(lam, "__len__"):
            raise ValueError("Grid solver accepts one lambda at a time")
        weights = (
            np.ones(self.nnodes, dtype=np.float64)
            if self.weights is None
            else self.weights
        )
        steps = self.workspace.solve(
            self.y,
            weights,
            self.beta,
            self.z,
            float(lam),
            self.maxsteps,
            self.converge,
        )
        self.steps.append(steps)
        return self.beta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True)
    parser.add_argument("--candidate-x", required=True)
    parser.add_argument("--candidate-til", required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--out-log-r", required=True)
    parser.add_argument("--out-log-rtil", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-sweeps", type=int, default=10)
    parser.add_argument("--fdr-level", type=float, default=0.1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--pr-table-size", type=int, default=524288)
    parser.add_argument(
        "--trail-mode", choices=("greedy", "axis"), default="greedy"
    )
    parser.add_argument(
        "--solver-mode",
        choices=("official", "persistent", "grid_primal_dual"),
        default="official",
    )
    parser.add_argument("--gfl-threads", type=int, default=1)
    parser.add_argument("--grid-maxsteps", type=int, default=400)
    parser.add_argument("--grid-converge", type=float, default=1e-5)
    parser.add_argument("--lambda-min", type=float, default=0.20)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-bins", type=int, default=30)
    parser.add_argument(
        "--trail-cache",
        default=os.environ.get("SEFT_FDRS_TRAIL_CACHE", str(Path.cwd() / ".cache/smoothfdr_trails")),
    )
    return parser.parse_args()


def read_volume(path, dims):
    values = np.fromfile(path, dtype="<f8")
    if values.size != int(np.prod(dims)):
        raise ValueError(f"{path}: expected {np.prod(dims)} values")
    return values.reshape(dims, order="F")


def write_volume(path, volume):
    np.asarray(volume, dtype="<f8").ravel(order="F").tofile(path)


def numpy_sweep_order(size, sweeps, seed):
    np.random.seed(seed)
    pieces = []
    for _ in range(sweeps):
        current = np.arange(size, dtype=np.int64)
        np.random.shuffle(current)
        pieces.append(current)
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.int64)


def _edge_map(edge_array):
    result = defaultdict(list)
    for left, right in np.asarray(edge_array, dtype=np.int32):
        result[int(left)].append(int(right))
        result[int(right)].append(int(left))
    return result


def load_or_create_trails(dims, cache_root, verbose=0):
    """Cache the official deterministic greedy decomposition by lattice size."""
    from networkx import Graph
    from pygfl.trails import decompose_graph
    from pygfl.utils import chains_to_trails, hypercube_edges

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = "x".join(str(value) for value in dims)
    cache_file = cache_root / f"greedy_{stem}.npz"
    lock_file = cache_root / f"greedy_{stem}.lock"
    with open(lock_file, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cache_file.exists():
            if verbose:
                print(f"Creating official greedy-trail cache {cache_file}")
            edges = np.asarray(hypercube_edges(dims), dtype=np.int32)
            graph = Graph()
            graph.add_edges_from(edges)
            chains = decompose_graph(graph, heuristic="greedy")
            ntrails, trails, breakpoints, _ = chains_to_trails(chains)
            temporary = cache_file.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                edges=edges,
                trails=np.asarray(trails, dtype=np.int32),
                breakpoints=np.asarray(breakpoints, dtype=np.int32),
                ntrails=np.asarray([ntrails], dtype=np.int32),
            )
            temporary.replace(cache_file)
        cached = np.load(cache_file)
        return (
            int(cached["ntrails"][0]),
            np.ascontiguousarray(cached["trails"], dtype=np.int32),
            np.ascontiguousarray(cached["breakpoints"], dtype=np.int32),
            np.ascontiguousarray(cached["edges"], dtype=np.int32),
        )


def load_or_create_axis_trails(dims, cache_root, verbose=0):
    """Decompose a rectangular grid into one straight trail per axis-line.

    Every grid edge occurs exactly once, so this represents the same graph-TV
    objective as the greedy decomposition while avoiding hundreds of
    thousands of very short trails on a 64^3 grid.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = "x".join(str(value) for value in dims)
    cache_file = cache_root / f"axis_{stem}.npz"
    lock_file = cache_root / f"axis_{stem}.lock"
    with open(lock_file, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cache_file.exists():
            if verbose:
                print(f"Creating axis-trail cache {cache_file}")
            nodes = np.arange(np.prod(dims), dtype=np.int32).reshape(dims)
            trail_parts = []
            breakpoint_parts = []
            edge_parts = []
            offset = 0
            for axis, length in enumerate(dims):
                lines = np.moveaxis(nodes, axis, -1).reshape(-1, length)
                trail_parts.append(lines.ravel())
                offset_part = offset + np.arange(
                    1, lines.shape[0] + 1, dtype=np.int64
                ) * length
                breakpoint_parts.append(offset_part.astype(np.int32))
                offset = int(offset_part[-1])
                edge_parts.append(
                    np.column_stack(
                        (lines[:, :-1].ravel(), lines[:, 1:].ravel())
                    )
                )
            trails = np.ascontiguousarray(
                np.concatenate(trail_parts), dtype=np.int32
            )
            breakpoints = np.ascontiguousarray(
                np.concatenate(breakpoint_parts), dtype=np.int32
            )
            edges = np.ascontiguousarray(
                np.concatenate(edge_parts), dtype=np.int32
            )
            temporary = cache_file.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                edges=edges,
                trails=trails,
                breakpoints=breakpoints,
                ntrails=np.asarray([breakpoints.size], dtype=np.int32),
            )
            temporary.replace(cache_file)
        cached = np.load(cache_file)
        return (
            int(cached["ntrails"][0]),
            np.ascontiguousarray(cached["trails"], dtype=np.int32),
            np.ascontiguousarray(cached["breakpoints"], dtype=np.int32),
            np.ascontiguousarray(cached["edges"], dtype=np.int32),
        )


def main():
    args = parse_args()
    dims = tuple(int(value) for value in args.dims.split(","))
    if len(dims) != 3:
        raise ValueError("--dims must have three comma-separated values")
    if args.num_sweeps != 10:
        raise ValueError("The official repository default is 10 sweeps")
    if (
        args.lambda_bins < 1
        or args.lambda_min <= 0
        or args.lambda_max < args.lambda_min
    ):
        raise ValueError("invalid lambda path")
    if not hasattr(np, "product"):
        np.product = np.prod

    from pygfl.solver import TrailSolver
    import smoothfdr.easy as smoothfdr_easy
    from smoothfdr.easy import solution_path_smooth_fdr

    background = read_volume(args.background, dims)
    candidate_x = read_volume(args.candidate_x, dims)
    candidate_til = read_volume(args.candidate_til, dims)
    flat = np.ascontiguousarray(background.ravel(), dtype=np.float64)
    started = time.time()

    mean_grid = np.linspace(
        max(-20.0, float(flat.min()) - 1.0),
        min(20.0, float(flat.max()) + 1.0),
        220,
    )
    sweep_order = numpy_sweep_order(
        flat.size, args.num_sweeps, args.seed
    )
    pr_started = time.time()
    pr = absmax_backend.predictive_recursion_absmax(
        flat,
        np.ascontiguousarray(mean_grid),
        sweep_order,
        -0.67,
        args.pr_table_size,
    )
    pr_seconds = time.time() - pr_started
    null_dist = AbsMaxNullDistribution()
    signal_background_dist = InterpolatedDistribution(
        pr["grid"], pr["g1"]
    )

    trails_started = time.time()
    if args.solver_mode == "grid_primal_dual":
        ntrails = 0
        trails = np.empty(0, dtype=np.int32)
        breakpoints = np.empty(0, dtype=np.int32)
        edge_array = None
        solver = GridPrimalDualSolver(
            dims,
            maxsteps=args.grid_maxsteps,
            converge=args.grid_converge,
            gfl_threads=args.gfl_threads,
        )
    else:
        trail_loader = (
            load_or_create_trails
            if args.trail_mode == "greedy"
            else load_or_create_axis_trails
        )
        ntrails, trails, breakpoints, edge_array = trail_loader(
            dims, args.trail_cache, args.verbose
        )
        solver = (
            TrailSolver()
            if args.solver_mode == "official"
            else PersistentTrailSolver(gfl_threads=args.gfl_threads)
        )
    trail_seconds = time.time() - trails_started
    solver.set_data(
        flat,
        None if edge_array is None else _edge_map(edge_array),
        ntrails,
        trails,
        breakpoints,
    )
    smoothing_started = time.time()
    original_calc_plateaus = smoothfdr_easy.calc_plateaus
    if args.solver_mode in ("persistent", "grid_primal_dual"):
        smoothfdr_easy.calc_plateaus = lambda beta, _edges: range(
            solver.workspace.plateau_count(
                np.ascontiguousarray(beta, dtype=np.float64), 1e-4
            )
        )
    try:
        fit = solution_path_smooth_fdr(
            flat,
            solver,
            null_dist,
            signal_background_dist,
            min_lambda=args.lambda_min,
            max_lambda=args.lambda_max,
            lambda_bins=args.lambda_bins,
            verbose=max(0, args.verbose - 1),
        )
    finally:
        smoothfdr_easy.calc_plateaus = original_calc_plateaus
    smoothing_seconds = time.time() - smoothing_started
    background_signal = np.asarray(
        fit["posteriors"], dtype=np.float64
    ).reshape(dims)
    grid = np.asarray(pr["grid"], dtype=np.float64)
    f1 = np.asarray(pr["f1"], dtype=np.float64)
    g1 = np.asarray(pr["g1"], dtype=np.float64)
    reweight_started = time.time()
    score_x = absmax_backend.reweight_scores(
        np.ascontiguousarray(background.ravel()),
        np.ascontiguousarray(candidate_x.ravel()),
        np.ascontiguousarray(background_signal.ravel()),
        grid,
        f1,
        g1,
    )
    score_til = absmax_backend.reweight_scores(
        np.ascontiguousarray(background.ravel()),
        np.ascontiguousarray(candidate_til.ravel()),
        np.ascontiguousarray(background_signal.ravel()),
        grid,
        f1,
        g1,
    )
    reweight_seconds = time.time() - reweight_started
    log_r = np.asarray(score_x["log_score"]).reshape(dims)
    log_rtil = np.asarray(score_til["log_score"]).reshape(dims)
    write_volume(args.out_log_r, log_r)
    write_volume(args.out_log_rtil, log_rtil)

    diagnostics = {
        "backend": "smoothfdr official GFL with C++ abs-max emissions",
        "adaptation": "candidate f0/f1 and baseline g0/g1",
        "smoothfdr_version": "0.9.5",
        "num_sweeps": args.num_sweeps,
        "predictive_recursion_grid_size": 220,
        "predictive_recursion_likelihood_table_size": args.pr_table_size,
        "fdr_level_fit_only": args.fdr_level,
        "seed": args.seed,
        "lambda": float(fit["lambda"]),
        "lambda_index": int(fit["best"]),
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "lambda_bins": args.lambda_bins,
        "pi0_predictive_recursion": float(pr["pi0"]),
        "ntrails": ntrails,
        "trail_mode": (
            "none"
            if args.solver_mode == "grid_primal_dual"
            else args.trail_mode
        ),
        "solver_mode": args.solver_mode,
        "gfl_threads": args.gfl_threads,
        "grid_maxsteps": args.grid_maxsteps,
        "grid_converge": args.grid_converge,
        "gfl_solve_calls": len(solver.steps),
        "gfl_steps_total": int(np.sum(solver.steps)),
        "gfl_steps_mean": float(np.mean(solver.steps)),
        "gfl_steps_max": int(np.max(solver.steps)),
        "trail_cache": (
            None
            if args.solver_mode == "grid_primal_dual"
            else str(
                Path(args.trail_cache)
                / (
                    f"{args.trail_mode}_"
                    f"{'x'.join(str(value) for value in dims)}.npz"
                )
            )
        ),
        "prior_signal_range": [
            float(np.min(fit["priors"])),
            float(np.max(fit["priors"])),
        ],
        "posterior_signal_range": [
            float(np.min(fit["posteriors"])),
            float(np.max(fit["posteriors"])),
        ],
        "elapsed_seconds": time.time() - started,
        "predictive_recursion_seconds": pr_seconds,
        "trail_setup_seconds": trail_seconds,
        "smoothing_path_seconds": smoothing_seconds,
        "candidate_reweight_seconds": reweight_seconds,
    }
    with open(args.diagnostics, "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)


if __name__ == "__main__":
    main()
