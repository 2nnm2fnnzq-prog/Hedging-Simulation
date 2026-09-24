"""Independent pricing, Greek, IV and quadrature checks; exits nonzero on failure."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

try:
    from .pricing import heston_price_greeks, quantlib_price, bs_iv_delta, bs_price, get_diagnostics
except ImportError:
    from pricing import heston_price_greeks, quantlib_price, bs_iv_delta, bs_price, get_diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"


def finite_differences(s, v, k, t, r, p):
    """Independent central/forward differences, Richardson-extrapolated.

    These deliberately use different bump sizes from the production fallback.
    Derivatives are with respect to variance v, not volatility sqrt(v).
    """
    hs = s * 2e-5
    hv = min(2e-5, max(1e-6, 0.015 * (v + p["kappa"] * p["theta"] * t / 2)))

    def ds(h):
        return (quantlib_price(s+h, v, k, t, r, p) - quantlib_price(s-h, v, k, t, r, p)) / (2*h)

    def dv(h):
        if v >= h:
            return (quantlib_price(s, v+h, k, t, r, p) - quantlib_price(s, v-h, k, t, r, p)) / (2*h)
        return (-3*quantlib_price(s, v, k, t, r, p) + 4*quantlib_price(s, v+h, k, t, r, p) - quantlib_price(s, v+2*h, k, t, r, p)) / (2*h)

    return (4*ds(hs/2)-ds(hs))/3, (4*dv(hv/2)-dv(hv))/3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    base = dict(kappa=2.75, theta=0.0834, xi=0.550, rho=-0.569)
    cases = [("reference", base)]
    for key in ("kappa", "theta", "xi", "rho"):
        for direction in (-1, 1):
            value = base[key] + direction*0.1 if key == "rho" else base[key]*(1+direction*0.2)
            cases.append((f"{key}_{direction:+d}", dict(base, **{key: value})))
    rows = []
    get_diagnostics(reset=True)
    for case, parameters in cases:
        for tau in (1/252, 5/252, 63/252, 1.0):
            s = np.tile([60., 90., 100., 110., 160.], 5)
            v = np.repeat([0., 0.001, 0.01, 0.0483, 0.3], 5)
            values = heston_price_greeks(s, v, 100.0, tau, 0.04, parameters)
            more_nodes = heston_price_greeks(s, v, 100.0, tau, 0.04, parameters, quadrature_order=180)
            for j, (spot, variance) in enumerate(zip(s, v)):
                ref = quantlib_price(spot, variance, 100.0, tau, 0.04, parameters)
                d, dv = finite_differences(spot, variance, 100.0, tau, 0.04, parameters)
                rows.append(dict(scenario=case, spot=spot, variance=variance, tau=tau,
                                 price=float(values[0][j]), delta=float(values[1][j]), variance_derivative=float(values[2][j]),
                                 quantlib_price=ref, finite_difference_delta=d, finite_difference_variance=dv,
                                 price_error=abs(values[0][j]-ref), delta_error=abs(values[1][j]-d), variance_derivative_error=abs(values[2][j]-dv),
                                 quadrature_price_difference=abs(values[0][j]-more_nodes[0][j]),
                                 quadrature_delta_difference=abs(values[1][j]-more_nodes[1][j]),
                                 quadrature_variance_difference=abs(values[2][j]-more_nodes[2][j])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pricing_cases.csv", index=False)
    tolerances = dict(price_error=3e-7, delta_error=1e-5, variance_derivative_error=2e-3,
                      quadrature_price_difference=3e-7, quadrature_delta_difference=1e-5, quadrature_variance_difference=2e-3)
    checks = {column: dict(maximum=float(df[column].max()), tolerance=tolerance,
                          passed=bool(df[column].max() <= tolerance)) for column, tolerance in tolerances.items()}
    # Round-trip inversion is tested away from numerical zero-time-value limits.
    s = np.array([70., 90., 100., 110., 140.])
    sigma = np.array([0.20, 0.25, 0.30, 0.35, 0.4])
    c = bs_price(s, 100.0, 0.5, 0.04, sigma)
    iv, delta = bs_iv_delta(c, s, 100.0, 0.5, 0.04)
    checks["bs_iv_round_trip"] = dict(maximum=float(np.max(abs(iv-sigma))), tolerance=1e-10,
                                       passed=bool(np.max(abs(iv-sigma)) < 1e-10))
    # Analytic zero-dividend martingale prices approach BS when xi tends to zero.
    near_constant = dict(kappa=2.75, theta=0.0483, xi=1e-4, rho=0.0)
    c, _, _ = heston_price_greeks(np.array([90.,100.,110.]), np.full(3,0.0483), 100., 1., 0.04, near_constant)
    target = bs_price(np.array([90.,100.,110.]),100.,1.,0.04,np.sqrt(0.0483))
    checks["constant_variance_limit"] = dict(maximum=float(np.max(abs(c-target))), tolerance=1e-5,
                                              passed=bool(np.max(abs(c-target)) < 1e-5))
    passed = all(check["passed"] for check in checks.values())
    report = dict(passed=passed, grid_cases=len(df), scenarios=len(cases), checks=checks,
                  diagnostics=get_diagnostics(), seconds=time.perf_counter()-start,
                  notes=["Prices independently checked with QuantLib adaptive integration.",
                         "Greeks checked against Richardson differences using distinct bumps; variance derivative is not volatility vega.",
                         "Quadrature comparison uses checked 144-node and checked 180-node engines, each with its counted adaptive fallback.",
                         "Variance zero in QuantLib is regularized to 1e-12; production Fourier formula allows exactly zero.",
                         "The variance-Greek tolerance of 0.002 maps to at most about 1.5e-5 shares under the tested spot/parameter grid."])
    (OUT / "pricing.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
