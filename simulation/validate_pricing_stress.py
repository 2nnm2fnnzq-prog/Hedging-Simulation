"""Reproduce the bounded 640-state Heston pricing stress audit.

This is a numerical check, not an additional hedging experiment. The fixed
states intentionally extend beyond the ordinary validation grid. No market
paths, study settings, hedge outcomes, or core pricing modules are changed.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

try:
    from .pricing import heston_price_greeks, quantlib_price, get_diagnostics
except ImportError:
    from pricing import heston_price_greeks, quantlib_price, get_diagnostics


def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed = 24791
    rng = np.random.default_rng(seed)
    # Independent log-uniform state draws; identical states at each expiry.
    spots = np.exp(rng.uniform(np.log(25.0), np.log(300.0), 128))
    variances = np.exp(rng.uniform(np.log(1e-6), np.log(0.6), 128))
    expiries = [1/252, 0.02, 0.1, 0.5, 1.0]
    params = dict(kappa=2.75, theta=0.0834, xi=0.55, rho=-0.569)
    tolerance = 3e-7
    rows = []
    get_diagnostics(reset=True)
    for tau in expiries:
        price, delta, variance_derivative = heston_price_greeks(
            spots, variances, 100.0, tau, 0.04, params
        )
        reference = np.array([
            quantlib_price(s, v, 100.0, tau, 0.04, params)
            for s, v in zip(spots, variances)
        ])
        for i in range(len(spots)):
            rows.append(dict(
                state_id=i, spot=float(spots[i]), variance=float(variances[i]), tau=tau,
                price=float(price[i]), delta=float(delta[i]),
                variance_derivative=float(variance_derivative[i]),
                quantlib_price=float(reference[i]),
                absolute_price_error=float(abs(price[i]-reference[i])),
            ))
    frame = pd.DataFrame(rows)
    finite = bool(np.isfinite(frame.select_dtypes(include="number").to_numpy()).all())
    maximum = float(frame.absolute_price_error.max())
    report = dict(
        passed=finite and maximum <= tolerance,
        seed=seed, state_draws=128, expiries=expiries, price_comparisons=len(frame),
        state_distribution=dict(spot="log-uniform on [25,300]", variance="log-uniform on [1e-6,0.6]"),
        parameters=params, strike=100.0, rate=0.04,
        all_outputs_finite=finite, maximum_absolute_price_error=maximum,
        price_tolerance=tolerance,
        maximum_error_by_expiry={str(t): float(v) for t, v in frame.groupby("tau").absolute_price_error.max().items()},
        diagnostics=get_diagnostics(), seconds=time.perf_counter()-started,
        notes=[
            "Additional numerical state coverage; no new hedging outcomes or research scenarios.",
            "Prices compared with independent QuantLib adaptive integration.",
            "Delta and variance-derivative finiteness are checked here; their accuracy is assessed separately by validate_pricing.py.",
            "Production adaptive fallbacks and roundoff adjustments remain enabled and counted; no states excluded.",
        ],
    )
    frame.to_csv(out / "pricing_stress_cases.csv", index=False)
    (out / "pricing_stress.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
