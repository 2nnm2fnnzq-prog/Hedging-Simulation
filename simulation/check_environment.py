"""Deterministic software checks; no market simulation or research results."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_checks() -> dict:
    cache = Path(__file__).resolve().parents[1] / "outputs" / "setup" / "matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    import numpy as np
    import pandas as pd
    import QuantLib as ql
    from scipy.optimize import brentq
    from scipy.stats import norm

    packages = ["numpy", "scipy", "pandas", "matplotlib", "jupyterlab", "QuantLib"]
    versions = {name: importlib.metadata.version(name) for name in packages}

    # A standard diagnostic contract, deliberately unrelated to the study baseline.
    spot, strike, rate, vol, years = 100.0, 100.0, 0.05, 0.20, 1.0

    def bs_call(sigma: float) -> float:
        d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * years) / (
            sigma * math.sqrt(years)
        )
        d2 = d1 - sigma * math.sqrt(years)
        return float(spot * norm.cdf(d1) - strike * math.exp(-rate * years) * norm.cdf(d2))

    reference = bs_call(vol)
    recovered_vol = brentq(lambda sigma: bs_call(sigma) - reference, 0.01, 2.0)
    checks = {
        "numpy_array_arithmetic": bool(np.sum(np.array([1.0, 2.0, 3.0])) == 6.0),
        "pandas_table": bool(pd.DataFrame({"a": [1, 2]}).a.sum() == 3),
        "black_scholes_known_value": abs(reference - 10.450583572185565) < 1e-10,
        "scipy_iv_round_trip": abs(recovered_vol - vol) < 1e-10,
    }

    old_date = ql.Settings.instance().evaluationDate
    try:
        today = ql.Date(1, 1, 2025)
        maturity = ql.Date(1, 1, 2026)
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        risk_free = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count))
        dividends = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
        stock = ql.QuoteHandle(ql.SimpleQuote(spot))
        volatility = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), vol, day_count)
        )
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, strike), ql.EuropeanExercise(maturity)
        )
        bs_process = ql.BlackScholesMertonProcess(stock, dividends, risk_free, volatility)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))
        ql_bs = float(option.NPV())
        checks["quantlib_bs_crosscheck"] = abs(ql_bs - reference) < 1e-9

        # Constant-variance limiting case: small xi and v0 == theta == vol**2.
        process = ql.HestonProcess(risk_free, dividends, stock, vol**2, 1.0, vol**2, 1e-4, 0.0)
        option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 128))
        ql_heston = float(option.NPV())
        checks["quantlib_heston_constant_variance_limit"] = abs(ql_heston - reference) < 2e-4
    finally:
        ql.Settings.instance().evaluationDate = old_date

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    checks["matplotlib_figure_creation"] = True

    return {
        "kind": "software_smoke_check",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "Deterministic library checks only; no simulated paths or hedge rankings.",
        "research_engine_validated": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "checks": checks,
        "diagnostic_values": {
            "independent_bs_call": reference,
            "quantlib_bs_call": ql_bs,
            "quantlib_heston_near_constant_variance_call": ql_heston,
            "recovered_implied_volatility": recovered_vol,
        },
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "setup"
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = run_checks()
    except Exception as exc:
        report = {
            "kind": "software_smoke_check",
            "status": "failed",
            "research_engine_validated": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    (output / "environment_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
