"""Check path implementation, moments and coupled time-grid sensitivity.

This check does not establish convergence of the eventual hedge losses.
That separate check must use each coupled path grid through the whole ledger.
Run from the repository root: .venv/bin/python simulation/validate_paths.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .paths import generate_paths
except ImportError:
    from paths import generate_paths


REFERENCE_P = {"mu": 0.10, "kappa": 4.75, "theta": 0.0483, "xi": 0.550, "rho": -0.569}


def moment_check(sample: np.ndarray, expected: float) -> dict:
    mean = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    z = (mean - expected) / se if se else (0.0 if mean == expected else math.inf)
    return {"estimate": mean, "continuous_model_value": expected, "monte_carlo_se": se,
            "standardized_difference": z, "within_four_mc_standard_errors": abs(z) <= 4.0}


def validate(n_paths: int = 16384, days: int = 252, seed: int = 18092026) -> dict:
    # These moment checks use the continuous CIR expressions, so discrepancies
    # can reflect discretization bias as well as Monte Carlo variability.
    results = {"path_count": n_paths, "days": days, "seed": seed,
               "market_parameters": REFERENCE_P, "checks": {}, "grids": {}}
    checks = results["checks"]
    deterministic = dict(REFERENCE_P, xi=0.0)
    s2, v2, _ = generate_paths(32, 10, 2, seed, deterministic, coupling_substeps=8)
    s8, v8, _ = generate_paths(32, 10, 8, seed, deterministic, coupling_substeps=8)
    checks["constant_variance_coupling"] = bool(
        np.max(np.abs(s2 - s8)) < 1e-10
        and np.array_equal(v2, v8)
        and np.all(v8 == REFERENCE_P["theta"])
    )
    s_again, v_again, _ = generate_paths(32, 10, 8, seed, deterministic, coupling_substeps=8)
    checks["exact_seed_reproducibility"] = bool(np.array_equal(s8, s_again) and np.array_equal(v8, v_again))
    terminal = {}
    t = days / 252
    theta, kappa, xi = (REFERENCE_P[k] for k in ("theta", "kappa", "xi"))
    decay = math.exp(-kappa * t)
    expected_var_mean = theta
    expected_var_variance = theta * xi**2 * (1.0 - decay**2) / (2.0 * kappa)
    for substeps in (2, 4, 8):
        stock, variance, diagnostics = generate_paths(n_paths, days, substeps, seed, REFERENCE_P, coupling_substeps=8)
        terminal[substeps] = (stock[:, -1].copy(), variance[:, -1].copy())
        results["grids"][str(substeps)] = {
            "diagnostics": diagnostics,
            "stock_first_moment": moment_check(stock[:, -1], 100.0 * math.exp(REFERENCE_P["mu"] * t)),
            "variance_first_moment": moment_check(variance[:, -1], expected_var_mean),
            "variance_second_raw_moment": moment_check(variance[:, -1]**2, expected_var_variance + expected_var_mean**2),
        }
    finest = terminal[8]
    paired = {}
    for substeps in (2, 4):
        stock_difference = terminal[substeps][0] - finest[0]
        variance_difference = terminal[substeps][1] - finest[1]
        paired[str(substeps)] = {
            "stock_endpoint_rmse_vs_eight_substeps": float(np.sqrt(np.mean(stock_difference**2))),
            "variance_endpoint_rmse_vs_eight_substeps": float(np.sqrt(np.mean(variance_difference**2))),
            "stock_mean_difference_vs_eight_substeps": moment_check(stock_difference, 0.0),
            "variance_mean_difference_vs_eight_substeps": moment_check(variance_difference, 0.0),
        }
    results["paired_grid_comparisons"] = paired
    checks["stock_endpoint_error_decreases_toward_finest_grid"] = (
        paired["4"]["stock_endpoint_rmse_vs_eight_substeps"] < paired["2"]["stock_endpoint_rmse_vs_eight_substeps"]
    )
    checks["variance_endpoint_error_decreases_toward_finest_grid"] = (
        paired["4"]["variance_endpoint_rmse_vs_eight_substeps"] < paired["2"]["variance_endpoint_rmse_vs_eight_substeps"]
    )
    for label in ("stock_first_moment", "variance_first_moment", "variance_second_raw_moment"):
        checks[f"finest_grid_{label}_within_four_mc_standard_errors"] = results["grids"]["8"][label]["within_four_mc_standard_errors"]
    checks["no_invalid_or_excluded_paths"] = all(
        grid["diagnostics"]["invalid_path_count"] == grid["diagnostics"]["excluded_path_count"] == 0
        for grid in results["grids"].values()
    )
    results["all_checks_passed"] = bool(all(checks.values()))
    results["interpretation"] = (
        "Moment checks use four Monte Carlo standard errors as diagnostic thresholds, not formal hypothesis tests. "
        "They do not prove absence of discretization bias. Paired endpoint errors compare to the finest available grid, "
        "not to exact sample paths. Full hedge-outcome convergence remains a separate required check. "
        "Negative latent variance states are handled by the documented full-truncation scheme and retained."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=16384)
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--seed", type=int, default=18092026)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/path_validation.json"))
    args = parser.parse_args()
    report = validate(args.paths, args.days, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": report["checks"], "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["all_checks_passed"] else 1)
