#!/usr/bin/env python3
"""PLIS adapters for the authors' DeepFDR and fcHMRF-LIS implementations.

External repositories are resolved from ``SEFT_ML_VENDOR_ROOT``.
Official mode keeps the published repository defaults and rejects unsupported
input shapes instead of silently generalizing them. Candidate reweighting and
binary R/Python I/O are the only PLIS-specific additions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy import ndimage, special, stats

from absmax_models import (
    FastAbsMaxKDE,
    TorchAbsMaxNull,
    absmax_pvalues,
    fit_weighted_kde,
    reweight_candidate,
)


def storey_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Storey-Tibshirani q-values, matching the DeepFDR orientation reference."""
    from scipy.interpolate import UnivariateSpline

    original = np.asarray(p_values).ravel()
    order = np.argsort(original)
    reverse = np.argsort(order)
    ordered = original[order]
    m = ordered.size
    kappa = np.arange(0, 0.96, 0.01)
    estimates = np.asarray([
        np.sum(ordered > value) / (m * (1 - value)) for value in kappa
    ])
    pi0 = float(np.clip(UnivariateSpline(kappa, estimates, k=3)(1.0), 0, 1))
    q_values = np.empty(m)
    q_values[-1] = pi0 * ordered[-1]
    for index in range(m - 2, -1, -1):
        q_values[index] = min(
            pi0 * m * ordered[index] / (index + 1), q_values[index + 1]
        )
    return q_values[reverse]


def lis_decisions(lis: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    values = np.asarray(lis).ravel()
    order = np.argsort(values)
    cumulative_mean = np.cumsum(values[order]) / np.arange(1, values.size + 1)
    allowed = np.flatnonzero(cumulative_mean <= alpha)
    count = int(allowed[-1] + 1) if allowed.size else 0
    decisions = np.zeros(values.size, dtype=bool)
    decisions[order[:count]] = True
    return decisions


def dice_coefficient(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.sum(left) + np.sum(right)
    if denominator == 0:
        return 1.0
    return float(2 * np.sum(left & right) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "deepfdr",
            "fchmrf",
            "deepfdr_absmax",
            "fchmrf_absmax",
        ),
        required=True,
    )
    parser.add_argument("--background", required=True)
    parser.add_argument("--candidate-x", required=True)
    parser.add_argument("--candidate-til", required=True)
    parser.add_argument("--appearance")
    parser.add_argument("--dims", required=True)
    parser.add_argument("--out-r", required=True)
    parser.add_argument("--out-rtil", required=True)
    parser.add_argument("--out-log-r")
    parser.add_argument("--out-log-rtil")
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--em-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--kde-mode", choices=("fast", "exact"), default="fast")
    parser.add_argument("--kde-grid-size", type=int, default=16384)
    parser.add_argument("--kde-max-grid-size", type=int, default=65536)
    parser.add_argument("--kde-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--vendor-root",
        default=os.environ.get("SEFT_ML_VENDOR_ROOT", str(Path.cwd() / ".vendor")),
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # CUDA max_pool3d backward has no deterministic kernel in this PyTorch
        # release.  CPU audits are bit-reproducible; CUDA production runs keep
        # the official operator and emit a warning for that one limitation.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def read_volume(path: str, dims: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    if values.size != int(np.prod(dims)):
        raise ValueError(f"{path} has {values.size} values; expected {np.prod(dims)}")
    return values.reshape(dims, order="F")


def write_volume(path: str, volume: np.ndarray) -> None:
    np.asarray(volume, dtype="<f8").ravel(order="F").tofile(path)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def extract_patch(
    volume: np.ndarray,
    start: tuple[int, int, int],
    patch_size: int,
) -> tuple[np.ndarray, tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    source = tuple(
        slice(begin, min(begin + patch_size, size))
        for begin, size in zip(start, volume.shape)
    )
    patch = np.zeros((patch_size,) * 3, dtype=np.float32)
    target = tuple(slice(0, item.stop - item.start) for item in source)
    patch[target] = volume[source]
    return patch, source, target


def fit_deepfdr_context(
    background: np.ndarray,
    args: argparse.Namespace,
    *,
    absmax: bool = False,
) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as functional

    official_defaults = {
        "seed": 0,
        "epochs": 17,
        "channels": 64,
        "learning_rate": 1e-3,
    }
    simulation_runner_shape = background.shape == (30, 30, 30)
    if not simulation_runner_shape and any(size % 4 != 0 for size in background.shape):
        raise ValueError(
            "DeepFDR's two pooling layers require each unpadded dimension to "
            "be divisible by four in paper-scale mode; received "
            f"{background.shape}."
        )
    mismatches = []
    for key in ("seed", "epochs", "channels", "learning_rate"):
        observed = getattr(args, key)
        expected = official_defaults[key]
        if observed != expected:
            mismatches.append(f"{key}={observed} (official {expected})")
    if mismatches:
        raise ValueError(
            "Official DeepFDR mode does not accept changed training defaults: "
            + ", ".join(mismatches)
        )

    source_dir = Path(args.vendor_root) / "DeepFDR" / "src" / "DeepFDR"
    sys.path.insert(0, str(source_dir))
    model_module = load_module("deepfdr_official_model", source_dir / "model.py")
    util_module = load_module("deepfdr_official_util", source_dir / "util.py")

    if args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    network = model_module.WNet(1).to(device)

    def initialize(module):
        if "Conv" in module.__class__.__name__ and hasattr(module, "weight"):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )

    network.apply(initialize)
    optimizer_encoder = torch.optim.SGD(
        network.UEnc.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-5,
    )
    optimizer_whole = torch.optim.SGD(
        network.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-5,
    )

    normalized_cut = util_module.SoftNCutLoss3D()
    image = torch.from_numpy(background.astype(np.float32))[None, None].to(device)
    model_input = (
        functional.pad(image, (1, 1, 1, 1, 1, 1))
        if simulation_runner_shape
        else image
    )

    def encode():
        if simulation_runner_shape:
            return network(model_input, returns="enc")
        return torch.sigmoid(network.UEnc(model_input))

    def decode():
        if simulation_runner_shape:
            return torch.sigmoid(network(model_input, returns="dec"))
        encoding = torch.sigmoid(network.UEnc(model_input))
        return torch.sigmoid(network.UDec(encoding))

    def soft_ncut_loss(encoding):
        # The released utility sends its spatial mask to CUDA whenever a GPU
        # exists, even for an explicitly requested CPU run.  Restrict that
        # device check during the official loss call; no numerical formula is
        # changed.
        cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: device.type == "cuda"
        try:
            if simulation_runner_shape:
                return normalized_cut(image, encoding)
            # The released loss hard-codes 30^3 only in its forward() wrapper.
            # Its two public calculation methods already accept img_size.
            weights = normalized_cut.calculate_weights(
                image,
                batch_size=1,
                img_size=background.shape,
                ox=3,
                radius=3,
                oi=11,
            )
            associations = [
                normalized_cut.soft_n_cut_loss_single_k(
                    weights,
                    encoding[:, 0] if label == 0 else 1 - encoding[:, 0],
                    batch_size=1,
                    img_size=background.shape,
                    radius=3,
                )
                for label in range(2)
            ]
            return torch.mean(
                2 - torch.sum(torch.stack(associations), dim=0)
            )
        finally:
            torch.cuda.is_available = cuda_available

    normal = torch.distributions.Normal(0.0, 1.0)
    if absmax:
        abs_cdf = 2.0 * normal.cdf(torch.abs(image)) - 1.0
        p_value = 1.0 - abs_cdf**2
    else:
        p_value = 2.0 * (1.0 - normal.cdf(torch.abs(image)))
    loss_trace = []
    network.train()
    for _ in range(args.epochs):
        optimizer_encoder.zero_grad(set_to_none=True)
        encoding = encode()
        encoder_loss = soft_ncut_loss(encoding)
        encoder_loss.backward()
        optimizer_encoder.step()

        optimizer_whole.zero_grad(set_to_none=True)
        decoded = decode()
        decoder_loss = torch.mean((p_value - decoded) ** 2)
        decoder_loss.backward()
        optimizer_whole.step()
        loss_trace.append(
            {
                "encoder": float(encoder_loss.detach().cpu()),
                "decoder": float(decoder_loss.detach().cpu()),
            }
        )

    network.eval()
    with torch.no_grad():
        raw = (
            encode()
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    if absmax:
        reference = storey_qvalues(absmax_pvalues(background)) <= 0.1
    else:
        q_values = util_module.compute_qval(background, 0.1)
        reference = q_values <= 0.1
    k_raw, lis_raw = util_module.p_lis(raw, threshold=0.1, flip=False)
    k_flipped, lis_flipped = util_module.p_lis(raw, threshold=0.1, flip=True)
    discovery_raw_as_lis = np.zeros(raw.size, dtype=bool)
    discovery_flipped_as_lis = np.zeros(raw.size, dtype=bool)
    discovery_raw_as_lis[lis_raw[:k_raw]["index"]] = True
    discovery_flipped_as_lis[lis_flipped[:k_flipped]["index"]] = True
    dice_raw = util_module.dice(reference, discovery_raw_as_lis)
    dice_flipped = util_module.dice(reference, discovery_flipped_as_lis)
    flipped = bool(dice_raw < dice_flipped)
    lis = np.clip(1.0 - raw if flipped else raw, 1e-10, 1 - 1e-10)
    signal_probability = 1.0 - lis
    return signal_probability, {
        "backend": (
            "official DeepFDR simulation runner"
            if simulation_runner_shape
            else "DeepFDR paper-scale dimension generalization"
        ),
        "official_source_commit": "44294ac4742f1e2b1634158cfb869353c3708ea9",
        "device": str(device),
        "full_volume_training": True,
        "input_shape": list(background.shape),
        "padded_shape": (
            [32, 32, 32] if simulation_runner_shape else list(background.shape)
        ),
        "released_runner_exact": simulation_runner_shape,
        "dimension_handling": (
            "released 30^3 -> 32^3 zero padding"
            if simulation_runner_shape
            else "runtime img_size passed to official loss primitives"
        ),
        "epochs": args.epochs,
        "channels": args.channels,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "soft_ncut": "official util.SoftNCutLoss3D",
        "orientation_flipped": flipped,
        "dice_raw_as_lis": float(dice_raw),
        "dice_flipped_as_lis": float(dice_flipped),
        "qvalue_reference_discoveries": int(reference.sum()),
        "candidate_layer": (
            "PLIS local posterior-odds reweighting; not part of DeepFDR"
        ),
        "absmax_training_target": absmax,
        "orientation_reference": (
            "Storey q-values from abs-max null p-values"
            if absmax
            else "official normal-null q-values"
        ),
        "loss_trace": loss_trace,
    }


def plis_weighted_grid_density(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PLIS-only signal density used when a model has no emission density."""
    values = values.ravel()
    weights = np.maximum(weights.ravel(), 1e-8)
    weights /= weights.sum()
    bound = max(6.0, min(14.0, float(np.quantile(np.abs(values), 0.999)) + 1))
    grid = np.linspace(-bound, bound, 2049)
    edges = np.linspace(-bound, bound, 2050)
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    neff = (weights.sum() ** 2) / np.sum(weights**2)
    robust_sigma = min(
        float(np.std(values)),
        float(np.subtract(*np.percentile(values, [75, 25])) / 1.34),
    )
    robust_sigma = max(robust_sigma, 1e-3)
    bandwidth = max(0.9 * robust_sigma * neff ** (-0.2), 0.05)
    dx = grid[1] - grid[0]
    density = ndimage.gaussian_filter1d(
        counts / dx, sigma=max(bandwidth / dx, 0.5), mode="nearest"
    )
    density = np.maximum(density, 1e-12)
    density /= np.trapezoid(density, grid)
    return grid, density


def deepfdr_plis_candidate_null_scores(
    background: np.ndarray,
    candidate: np.ndarray,
    background_signal_probability: np.ndarray,
    grid: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Exchangeable local PLIS calibration of the official background map."""
    f1_background = np.maximum(
        np.interp(background.ravel(), grid, density), 1e-12
    )
    f1_candidate = np.maximum(
        np.interp(candidate.ravel(), grid, density), 1e-12
    )
    phi_background = np.maximum(stats.norm.pdf(background.ravel()), 1e-300)
    absmax_cdf = np.maximum(
        2.0 * special.ndtr(np.abs(background.ravel())) - 1.0, 1e-12
    )
    f0_background = np.maximum(2.0 * phi_background * absmax_cdf, 1e-300)
    f0_candidate = np.maximum(stats.norm.pdf(candidate.ravel()), 1e-300)
    q = np.clip(background_signal_probability.ravel(), 1e-6, 1 - 1e-6)
    log_odds = (
        np.log(q)
        - np.log1p(-q)
        + np.log(f1_candidate)
        - np.log(f0_candidate)
        - np.log(f1_background)
        + np.log(f0_background)
    )
    signal = special.expit(np.clip(log_odds, -35, 35))
    return np.clip(1.0 - signal, 1e-10, 1 - 1e-10).reshape(
        background.shape
    )


def fit_fchmrf_context(
    background: np.ndarray,
    appearance: np.ndarray,
    args: argparse.Namespace,
    *,
    absmax: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    import torch

    if not absmax and background.shape != (30, 30, 30):
        raise ValueError(
            "Official fcHMRF is hard-coded for one 30x30x30 volume; "
            f"received {background.shape}."
        )
    if args.em_steps != 5 or args.learning_rate != 1e-4 or args.threshold != 0.05:
        raise ValueError(
            "Official fcHMRF CLI defaults are em_steps=5, "
            "learning_rate=1e-4, threshold=0.05."
        )
    vendor = Path(args.vendor_root) / "fcHMRF-LIS"
    sys.path.insert(0, str(vendor))
    sys.path.insert(0, str(vendor / "src"))
    import src.model as official_model

    if absmax:
        FastAbsMaxKDE.default_mode = args.kde_mode
        FastAbsMaxKDE.initial_grid_size = args.kde_grid_size
        FastAbsMaxKDE.maximum_grid_size = args.kde_max_grid_size
        FastAbsMaxKDE.tolerance = args.kde_tolerance
        official_model.GaussianKDE = FastAbsMaxKDE
        official_model.Normal = TorchAbsMaxNull

    image = torch.from_numpy(background.astype(np.float32))[None, None]
    appearance_tensor = torch.from_numpy(
        appearance.astype(np.float32)
    )[None, None]
    network = official_model.fchmrf(
        image,
        appearance_tensor,
        mask=None,
        pval=None,
        threshold=args.threshold,
        lr=args.learning_rate,
    )
    if absmax:
        p_value = torch.from_numpy(
            absmax_pvalues(background).astype(np.float32)
        )[None, None]
        network.p_value = p_value
        network.p_mask = 1.0 - p_value
        network.h[:, 0:1] = p_value
        network.h[:, 1:2] = 1.0 - p_value
        network.kde.update_weights(
            network.h[:, 1:2].flatten(), update_kernel=True
        )
        network.f1_cont = network.kde.logpdf().reshape(network.I_shape)
    loss_trace = []
    for _ in range(args.em_steps):
        posterior, loss_q1, loss_q2 = network.em_step()
        loss_trace.append(
            {"q1": float(loss_q1.detach()), "q2": float(loss_q2.detach())}
        )
    network.eval()
    posterior, loss_q1, loss_q2 = network.em_step()
    loss_trace.append(
        {
            "q1": float(loss_q1.detach()),
            "q2": float(loss_q2.detach()),
            "final_repository_em_step": True,
        }
    )
    signal_probability = (
        posterior.squeeze().detach().cpu().numpy().astype(np.float64)
    )
    signal_probability = np.clip(signal_probability, 1e-4, 1 - 1e-4)
    if absmax:
        density_context = network.kde.context
    else:
        density_context = {
            "dataset": network.kde.dataset.detach().cpu().numpy().astype(np.float64),
            "weights": network.kde.weights.detach().cpu().numpy().astype(np.float64),
            "bandwidth": float(network.kde.bandwidth.detach().cpu()),
        }
    return signal_probability, density_context, {
        "backend": (
            "official fcHMRF CRF-RNN with C++ abs-max KDE"
            if absmax
            else "official fcHMRF including repository GaussianKDE"
        ),
        "official_source_commit": "f32e8550d40a83448525d811376f942873ec994a",
        "em_steps": args.em_steps,
        "final_repository_em_step": True,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
        "appearance_input": (
            "swap-invariant abs-max background adaptation"
            if absmax
            else "external delta_mu/beta map"
        ),
        "kde": (
            "C++ max-aware weighted grid KDE"
            if absmax
            else "official src.kde.GaussianKDE (quadratic kernel matrix)"
        ),
        "absmax_emissions": absmax,
        "loss_trace": loss_trace,
        "w0": float(network.w_0.detach()),
        "kde_bandwidth": (
            density_context.bandwidth
            if absmax
            else density_context["bandwidth"]
        ),
        "kde_diagnostics": (
            density_context.diagnostics() if absmax else None
        ),
    }


def official_weighted_gaussian_kde_pdf(
    evaluation: np.ndarray,
    density_context: dict,
    batch_size: int = 250,
) -> np.ndarray:
    """Evaluate the repository's weighted Gaussian KDE at arbitrary points."""
    dataset = density_context["dataset"].ravel()
    weights = density_context["weights"].ravel()
    weights = weights / weights.sum()
    bandwidth = density_context["bandwidth"]
    normalizer = bandwidth * math.sqrt(2.0 * math.pi)
    points = evaluation.ravel()
    result = np.empty(points.size, dtype=np.float64)
    for start in range(0, points.size, batch_size):
        stop = min(start + batch_size, points.size)
        differences = (points[start:stop, None] - dataset[None, :]) / bandwidth
        kernels = np.exp(-0.5 * differences**2) / normalizer
        result[start:stop] = kernels @ weights
    return np.maximum(result, 1e-12).reshape(evaluation.shape)


def candidate_null_scores(
    background: np.ndarray,
    candidate: np.ndarray,
    background_signal_probability: np.ndarray,
    density_context: dict,
    f1_background: np.ndarray | None = None,
) -> np.ndarray:
    if f1_background is None:
        f1_background = official_weighted_gaussian_kde_pdf(
            background, density_context
        )
    f1_background = f1_background.ravel()
    f1_candidate = official_weighted_gaussian_kde_pdf(
        candidate, density_context
    ).ravel()
    f0_background = np.maximum(stats.norm.pdf(background.ravel()), 1e-300)
    f0_candidate = np.maximum(stats.norm.pdf(candidate.ravel()), 1e-300)
    q = np.clip(background_signal_probability.ravel(), 1e-6, 1 - 1e-6)
    log_odds = (
        np.log(q)
        - np.log1p(-q)
        + np.log(f1_candidate)
        - np.log(f0_candidate)
        - np.log(f1_background)
        + np.log(f0_background)
    )
    signal = special.expit(np.clip(log_odds, -35, 35))
    return np.clip(1.0 - signal, 1e-10, 1 - 1e-10).reshape(
        background.shape
    )


def main() -> None:
    args = parse_args()
    dims = tuple(int(value) for value in args.dims.split(","))
    if len(dims) != 3:
        raise ValueError("--dims must contain three comma-separated integers")
    set_deterministic(args.seed)
    background = read_volume(args.background, dims)
    candidate_x = read_volume(args.candidate_x, dims)
    candidate_til = read_volume(args.candidate_til, dims)

    started = time.time()
    if args.method == "deepfdr":
        context, diagnostics = fit_deepfdr_context(background, args)
        grid, density = plis_weighted_grid_density(background, context)
        score_x = deepfdr_plis_candidate_null_scores(
            background, candidate_x, context, grid, density
        )
        score_til = deepfdr_plis_candidate_null_scores(
            background, candidate_til, context, grid, density
        )
        log_score_x = np.log(score_x)
        log_score_til = np.log(score_til)
    elif args.method == "deepfdr_absmax":
        context, diagnostics = fit_deepfdr_context(
            background, args, absmax=True
        )
        density_context = fit_weighted_kde(
            background,
            context,
            mode=args.kde_mode,
            initial_grid_size=args.kde_grid_size,
            maximum_grid_size=args.kde_max_grid_size,
            tolerance=args.kde_tolerance,
        )
        score_x, log_score_x = reweight_candidate(
            background, candidate_x, context, density_context
        )
        score_til, log_score_til = reweight_candidate(
            background, candidate_til, context, density_context
        )
        diagnostics.update(
            {
                "adaptation": "candidate f0/f1 and baseline g0/g1",
                "kde": density_context.diagnostics(),
            }
        )
    else:
        if args.appearance is None:
            raise ValueError(
                "Official fcHMRF requires --appearance with the delta_mu/beta map."
            )
        appearance = read_volume(args.appearance, dims)
        is_absmax = args.method == "fchmrf_absmax"
        context, density_context, diagnostics = fit_fchmrf_context(
            background, appearance, args, absmax=is_absmax
        )
        if is_absmax:
            score_x, log_score_x = reweight_candidate(
                background, candidate_x, context, density_context
            )
            score_til, log_score_til = reweight_candidate(
                background, candidate_til, context, density_context
            )
            diagnostics["adaptation"] = (
                "candidate f0/f1 and baseline g0/g1"
            )
        else:
            f1_background = official_weighted_gaussian_kde_pdf(
                background, density_context
            )
            score_x = candidate_null_scores(
                background, candidate_x, context, density_context, f1_background
            )
            score_til = candidate_null_scores(
                background, candidate_til, context, density_context, f1_background
            )
            log_score_x = np.log(score_x)
            log_score_til = np.log(score_til)
    write_volume(args.out_r, score_x)
    write_volume(args.out_rtil, score_til)
    if args.out_log_r:
        write_volume(args.out_log_r, log_score_x)
    if args.out_log_rtil:
        write_volume(args.out_log_rtil, log_score_til)
    diagnostics.update(
        {
            "method": args.method,
            "seed": args.seed,
            "elapsed_seconds": time.time() - started,
            "context_signal_range": [
                float(np.min(context)),
                float(np.max(context)),
            ],
            "finite_fraction": float(
                np.mean(np.isfinite(score_x) & np.isfinite(score_til))
            ),
        }
    )
    with open(args.diagnostics, "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)


if __name__ == "__main__":
    main()
