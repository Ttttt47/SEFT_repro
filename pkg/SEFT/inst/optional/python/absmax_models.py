"""Shared numerical utilities for the abs-max PLIS adaptations.

The expensive one-dimensional density operations live in ``absmax_backend``,
the local pybind11 extension.  This module supplies deterministic bandwidth
selection, adaptive grid validation, PyTorch-compatible fcHMRF objects, and
the common candidate/background posterior-odds reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import absmax_backend
except ImportError as error:  # pragma: no cover - diagnostic path
    raise ImportError(
        "The abs-max C++ backend is not built. Run "
        "SEFT/working_models/scripts/build_absmax_backend.sh."
    ) from error


def absmax_pvalues(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return np.asarray(
        absmax_backend.density_primitives(flat)["p_absmax"],
        dtype=np.float64,
    ).reshape(np.shape(values))


def silverman_bandwidth(values: np.ndarray, weights: np.ndarray) -> float:
    """Match the released fcHMRF weighted KDE bandwidth convention."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.maximum(np.asarray(weights, dtype=np.float64).ravel(), 1e-15)
    weights /= weights.sum()
    neff = float(1.0 / np.sum(weights**2))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    robust_scale = min(std, iqr / 1.34)
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = max(std, 1e-3)
    return max(0.9 * robust_scale * neff ** (-0.2), 1e-3)


@dataclass
class KDEContext:
    dataset: np.ndarray
    weights: np.ndarray
    bandwidth: float
    mode: str
    grid: np.ndarray | None
    f1: np.ndarray | None
    g1: np.ndarray | None
    grid_size: int | None
    max_log_error_f1: float | None
    max_log_error_g1: float | None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bandwidth": self.bandwidth,
            "grid_size": self.grid_size,
            "max_log_error_f1": self.max_log_error_f1,
            "max_log_error_g1": self.max_log_error_g1,
        }


def _validation_points(values: np.ndarray, count: int) -> np.ndarray:
    probabilities = np.linspace(0.001, 0.999, min(count, values.size))
    points = np.quantile(values, probabilities)
    # Explicitly probe both tails and small nonzero values.  The transformed
    # densities vanish at exactly zero, where relative log error is undefined.
    extra = np.asarray(
        [
            values.min(),
            values.max(),
            -1e-4,
            1e-4,
            -0.01,
            0.01,
        ],
        dtype=np.float64,
    )
    return np.unique(np.concatenate([points, extra]))


def fit_weighted_kde(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str = "fast",
    initial_grid_size: int = 16384,
    maximum_grid_size: int = 65536,
    tolerance: float = 1e-3,
    validation_points: int = 256,
) -> KDEContext:
    values = np.ascontiguousarray(values, dtype=np.float64).ravel()
    weights = np.maximum(
        np.ascontiguousarray(weights, dtype=np.float64).ravel(), 1e-15
    )
    if values.size != weights.size or not values.size:
        raise ValueError("values and weights must have equal nonzero length")
    weights /= weights.sum()
    bandwidth = silverman_bandwidth(values, weights)
    if mode == "exact":
        return KDEContext(
            dataset=values,
            weights=weights,
            bandwidth=bandwidth,
            mode=mode,
            grid=None,
            f1=None,
            g1=None,
            grid_size=None,
            max_log_error_f1=0.0,
            max_log_error_g1=0.0,
        )
    if mode != "fast":
        raise ValueError("mode must be 'fast' or 'exact'")

    probes = np.ascontiguousarray(
        _validation_points(values, validation_points), dtype=np.float64
    )
    exact = absmax_backend.exact_weighted_kde(
        values, weights, probes, bandwidth
    )
    grid_size = int(initial_grid_size)
    chosen = None
    error_f1 = error_g1 = math.inf
    while True:
        chosen = absmax_backend.fast_weighted_kde_grid(
            values, weights, bandwidth, grid_size
        )
        grid = np.asarray(chosen["grid"], dtype=np.float64)
        f1_fast = np.interp(probes, grid, chosen["f1"])
        g1_fast = np.interp(probes, grid, chosen["g1"])
        f1_exact = np.asarray(exact["f1"], dtype=np.float64)
        g1_exact = np.asarray(exact["g1"], dtype=np.float64)
        valid_f1 = f1_exact > 1e-12
        valid_g1 = (g1_exact > 1e-12) & (np.abs(probes) >= 1e-4)
        error_f1 = float(
            np.max(np.abs(np.log(f1_fast[valid_f1]) - np.log(f1_exact[valid_f1])))
        )
        error_g1 = float(
            np.max(np.abs(np.log(g1_fast[valid_g1]) - np.log(g1_exact[valid_g1])))
        )
        if max(error_f1, error_g1) <= tolerance:
            break
        if grid_size >= maximum_grid_size:
            raise RuntimeError(
                "Fast abs-max KDE failed its exact-density audit: "
                f"f1={error_f1:.3g}, g1={error_g1:.3g}, "
                f"grid_size={chosen['grid_size']}."
            )
        grid_size = min(maximum_grid_size, grid_size * 2)

    return KDEContext(
        dataset=values,
        weights=weights,
        bandwidth=bandwidth,
        mode=mode,
        grid=np.asarray(chosen["grid"], dtype=np.float64),
        f1=np.asarray(chosen["f1"], dtype=np.float64),
        g1=np.asarray(chosen["g1"], dtype=np.float64),
        grid_size=int(chosen["grid_size"]),
        max_log_error_f1=error_f1,
        max_log_error_g1=error_g1,
    )


def evaluate_context(
    context: KDEContext,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(points, dtype=np.float64).ravel()
    if context.mode == "exact":
        result = absmax_backend.exact_weighted_kde(
            context.dataset,
            context.weights,
            points,
            context.bandwidth,
        )
        return (
            np.asarray(result["f1"], dtype=np.float64),
            np.asarray(result["g1"], dtype=np.float64),
        )
    return (
        np.interp(points, context.grid, context.f1),
        np.interp(points, context.grid, context.g1),
    )


def reweight_candidate(
    background: np.ndarray,
    candidate: np.ndarray,
    background_signal: np.ndarray,
    context: KDEContext,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.shape(background)
    background_flat = np.ascontiguousarray(background, dtype=np.float64).ravel()
    candidate_flat = np.ascontiguousarray(candidate, dtype=np.float64).ravel()
    signal_flat = np.ascontiguousarray(
        background_signal, dtype=np.float64
    ).ravel()
    if context.mode == "fast":
        result = absmax_backend.reweight_scores(
            background_flat,
            candidate_flat,
            signal_flat,
            context.grid,
            context.f1,
            context.g1,
        )
        return (
            np.asarray(result["score"], dtype=np.float64).reshape(shape),
            np.asarray(result["log_score"], dtype=np.float64).reshape(shape),
        )

    f1_candidate, _ = evaluate_context(context, candidate_flat)
    _, g1_background = evaluate_context(context, background_flat)
    primitives_candidate = absmax_backend.density_primitives(candidate_flat)
    primitives_background = absmax_backend.density_primitives(background_flat)
    q = np.clip(signal_flat, 1e-12, 1 - 1e-12)
    log_odds = (
        np.log(q)
        - np.log1p(-q)
        + np.log(np.maximum(f1_candidate, 1e-300))
        - np.asarray(primitives_candidate["log_f0"])
        - np.log(np.maximum(g1_background, 1e-300))
        + np.asarray(primitives_background["log_g0"])
    )
    log_score = -np.logaddexp(0.0, log_odds)
    return (
        np.exp(np.maximum(log_score, -745.0)).reshape(shape),
        log_score.reshape(shape),
    )


class TorchAbsMaxNull:
    """Drop-in subset of ``torch.distributions.Normal`` used by fcHMRF."""

    def __init__(self, _mean, _sd):
        pass

    def log_prob(self, values):
        import torch

        normal = torch.distributions.Normal(
            torch.zeros((), dtype=values.dtype, device=values.device),
            torch.ones((), dtype=values.dtype, device=values.device),
        )
        a0 = (
            2.0 * normal.cdf(torch.abs(values)) - 1.0
        ).clamp_min(torch.finfo(values.dtype).tiny)
        return math.log(2.0) + normal.log_prob(values) + torch.log(a0)


class FastAbsMaxKDE:
    """fcHMRF-compatible KDE with max-aware background emissions."""

    default_mode = "fast"
    initial_grid_size = 16384
    maximum_grid_size = 65536
    tolerance = 1e-3
    validation_points = 256

    def __init__(self, dataset, batch_size=1000):
        import torch

        if dataset.ndim != 1:
            raise ValueError("Dataset must be one-dimensional")
        self.dataset = dataset.detach().clone()
        self.n = int(dataset.numel())
        self.batch_size = batch_size
        self.weights = torch.ones_like(dataset) / self.n
        self.context: KDEContext | None = None
        self.update_weights(self.weights, update_kernel=True)

    def update_weights(self, weights, update_kernel=True):
        import torch

        if int(weights.numel()) != self.n:
            raise ValueError("Weights must have the same length as dataset")
        normalized = weights.detach().flatten().clamp_min(1e-15)
        normalized = normalized / normalized.sum()
        self.weights = normalized
        self.neff = (normalized.sum() ** 2) / (normalized**2).sum()
        if update_kernel:
            self.context = fit_weighted_kde(
                self.dataset.detach().cpu().numpy(),
                normalized.detach().cpu().numpy(),
                mode=self.default_mode,
                initial_grid_size=self.initial_grid_size,
                maximum_grid_size=self.maximum_grid_size,
                tolerance=self.tolerance,
                validation_points=self.validation_points,
            )
            self.bandwidth = torch.tensor(
                self.context.bandwidth,
                dtype=self.dataset.dtype,
                device=self.dataset.device,
            )

    def logpdf(self, batch_size=None):
        import torch

        _, g1 = evaluate_context(
            self.context, self.dataset.detach().cpu().numpy()
        )
        return torch.from_numpy(np.log(np.maximum(g1, 1e-300))).to(
            dtype=self.dataset.dtype,
            device=self.dataset.device,
        )
