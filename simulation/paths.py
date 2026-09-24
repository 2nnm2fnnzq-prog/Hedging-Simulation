"""Reproducible Heston market paths under the physical measure.

The state carried between integration steps is a *latent* variance ``u``.
Full-truncation Euler uses ``v = max(u, 0)`` in both drift and diffusion:

    u_next = u + kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
    log(S_next / S) = (mu - v / 2) * dt + sqrt(v) * dW_s

with corr(dW_s, dW_v) = rho.  The latent state is not reset after a negative
step.  Only nonnegative ``v`` is supplied to option-pricing/hedging routines.
Daily observations include day zero and do not interpolate future values.

``coupling_substeps`` enables a common-Brownian-motion convergence check.
For identical seed, path count, days and coupling_substeps, the same fine
normal draws are aggregated to every coarser integration grid.  A different
path count is a different random draw layout: reproducibility is conditional
on all recorded configuration fields, not on a seed alone.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


def _parameter(params: Mapping[str, float] | Any, key: str) -> float:
    return float(params[key] if isinstance(params, Mapping) else getattr(params, key))


def generate_paths(
    n_paths: int,
    days: int,
    substeps: int,
    seed: int,
    params: Mapping[str, float] | Any,
    s0: float = 100.0,
    v0: float = 0.0483,
    trading_days: int = 252,
    coupling_substeps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return daily ``(stock, variance, diagnostics)`` for independent paths.

    ``params`` contains ``mu, kappa, theta, xi, rho`` in annual decimal units.
    Arrays have shape ``(n_paths, days + 1)``.  No failed paths are removed.
    Invalid inputs raise before simulation; nonfinite outputs raise with the
    failed count.  Boundary visits are retained and counted, not excluded.
    """
    for name, value in (("n_paths", n_paths), ("days", days),
                        ("substeps", substeps), ("trading_days", trading_days)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    finest = substeps if coupling_substeps is None else coupling_substeps
    if isinstance(finest, bool) or int(finest) != finest or finest < substeps or finest % substeps:
        raise ValueError("coupling_substeps must be a positive integer multiple of substeps")
    mu, kappa, theta, xi, rho = (
        _parameter(params, key) for key in ("mu", "kappa", "theta", "xi", "rho")
    )
    if not np.isfinite([mu, kappa, theta, xi, rho, s0, v0]).all():
        raise ValueError("All market parameters and initial states must be finite")
    if kappa <= 0 or theta <= 0 or xi < 0 or abs(rho) > 1 or s0 <= 0 or v0 < 0:
        raise ValueError("Require kappa, theta, s0 > 0; xi, v0 >= 0; abs(rho) <= 1")
    n_paths, days, substeps, trading_days, finest = map(
        int, (n_paths, days, substeps, trading_days, finest)
    )
    rng = np.random.default_rng(seed)
    dt = 1.0 / (trading_days * substeps)
    root_dt = math.sqrt(dt)
    independent_weight = math.sqrt(max(0.0, 1.0 - rho * rho))
    aggregation = finest // substeps
    stock = np.empty((n_paths, days + 1), dtype=np.float64)
    variance = np.empty_like(stock)
    stock[:, 0], variance[:, 0] = s0, v0
    log_stock = np.full(n_paths, math.log(s0), dtype=np.float64)
    latent = np.full(n_paths, v0, dtype=np.float64)
    negative_steps = 0
    affected_paths = np.zeros(n_paths, dtype=bool)
    minimum_latent = v0
    for day in range(1, days + 1):
        # Keep the random draw shape independent of the integration grid.
        fine_normals = rng.standard_normal((n_paths, finest, 2))
        normals = fine_normals.reshape(n_paths, substeps, aggregation, 2).sum(axis=2)
        normals /= math.sqrt(aggregation)
        for step in range(substeps):
            positive = np.maximum(latent, 0.0)
            root_v_dt = np.sqrt(positive) * root_dt
            z_variance = normals[:, step, 0]
            z_stock = rho * z_variance + independent_weight * normals[:, step, 1]
            log_stock += (mu - 0.5 * positive) * dt + root_v_dt * z_stock
            latent += kappa * (theta - positive) * dt + xi * root_v_dt * z_variance
            negative = latent < 0.0
            negative_steps += int(np.count_nonzero(negative))
            affected_paths |= negative
            minimum_latent = min(minimum_latent, float(latent.min()))
        stock[:, day] = np.exp(log_stock)
        variance[:, day] = np.maximum(latent, 0.0)
    failed = (~np.isfinite(stock).all(axis=1) | ~np.isfinite(variance).all(axis=1)
              | (stock <= 0.0).any(axis=1))
    failed_count = int(np.count_nonzero(failed))
    if failed_count:
        raise FloatingPointError(f"Path generation produced {failed_count} invalid paths; none excluded")
    return stock, variance, {
        "scheme": "full-truncation latent variance Euler; log-stock Euler",
        "rng": "NumPy default_rng / PCG64",
        "seed": int(seed),
        "n_paths": n_paths,
        "days": days,
        "trading_days": trading_days,
        "substeps_per_day": substeps,
        "coupling_substeps_per_day": finest,
        "integration_dt_years": dt,
        "latent_negative_step_count": negative_steps,
        "latent_negative_step_fraction": negative_steps / (n_paths * days * substeps),
        "paths_with_latent_negative_state": int(affected_paths.sum()),
        "daily_zero_variance_count_excluding_initial": int(np.count_nonzero(variance[:, 1:] == 0.0)),
        "minimum_latent_variance": minimum_latent,
        "invalid_path_count": failed_count,
        "excluded_path_count": 0,
    }
