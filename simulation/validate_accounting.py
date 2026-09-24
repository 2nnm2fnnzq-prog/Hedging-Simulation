"""Deterministic checks of cash, trade timing, and terminal-loss accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .hedging import hedge_paths
except ImportError:
    from hedging import hedge_paths


def run_validation() -> dict:
    checks = []

    def record(name: str, actual, expected, atol: float = 1e-11) -> None:
        difference = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=name)
        checks.append({"name": name, "status": "PASS", "max_absolute_residual": float(difference)})

    # A cash-only hedge earns interest, pays no fees, then pays the call payoff.
    S = np.array([[100, 103, 102, 110], [100, 98, 90, 95]], dtype=float)
    target = np.zeros_like(S)
    target[:, -1] = np.nan  # Unused maturity targets may be absent.
    err, fees, _ = hedge_paths(S, target, 100, 0.04, 10, 1, 7.5)
    record("cash-only financing and payoff", err,
           7.5 * np.exp(0.04 * 3 / 252) - np.maximum(S[:, -1] - 100, 0))
    record("cash-only hedge has no stock fees", fees, [0, 0])

    # A call that stays strictly ITM on these deterministic paths is replicated
    # by one share financed with a discounted-strike cash liability.
    S = np.array([[100, 105, 102, 108], [100, 110, 104, 101]], dtype=float)
    target = np.ones_like(S)
    target[:, -1] = 9_999  # There must be no extra expiry target trade.
    endowment = 100 - 90 * np.exp(-0.04 * 3 / 252)
    err, fees, ledger = hedge_paths(S, target, 90, 0.04, 0, 1, endowment, ledger_path=0)
    record("self-financing stock and discounted strike replicate payoff", err, [0, 0])
    record("zero-fee replication has zero fee sum", fees, [0, 0])
    record("expiry liquidates existing holdings only", ledger["trade"].to_numpy(), [1, 0, 0, -1])

    # Opening and closing a constant hedge pay exactly two fees. Fees are paid
    # from the endowment and already included in the returned terminal error.
    S = np.full((1, 5), 100.0)
    target = np.full_like(S, 0.4)
    err, fees, ledger = hedge_paths(S, target, 100, 0, 10, 1, 10, ledger_path=0)
    record("flat-stock opening and liquidation charges", fees, [0.08])
    record("fees reduce terminal cash exactly once", err, [9.92])
    record("opening cost comes from common endowment", ledger.loc[0, "portfolio"], 9.96)
    record("terminal holdings are zero", ledger.loc[4, "holdings"], 0)

    # Independent closed-form accounting for two-day rebalancing. Changes in
    # supplied targets on nontrade dates must not execute any stock trade.
    S = np.array([[100, 101, 110, 105, 120]], dtype=float)
    target = np.array([[0.5, 0.8, 0.2, 0.9, 1_000]], dtype=float)
    E, r, f = 20.0, 0.04, 0.001
    err, fees, ledger = hedge_paths(S, target, 100, r, f * 10_000, 2, E, ledger_path=0)
    g = np.exp(r / 252)
    opening_cash = E - 0.5 * 100 - 0.5 * 100 * f
    middle_cash_flow = -(-0.3 * 110 + 0.3 * 110 * f)
    closing_cash_flow = -(-0.2 * 120 + 0.2 * 120 * f)
    expected_cash = opening_cash * g**4 + middle_cash_flow * g**2 + closing_cash_flow
    record("two-day schedule matches independent cash expression", err, [expected_cash - 20])
    record("only scheduled trades and terminal liquidation execute", ledger["trade"], [0.5, 0, -0.3, 0, -0.2])
    record("fee total contains each executed charge once", fees, [f * (50 + 33 + 24)])
    record("ledger cumulative fees match returned fee total", ledger.iloc[-1]["cumulative_fees"], fees[0])
    record("terminal payoff is deducted once", ledger.iloc[-1]["terminal_error"], err[0])
    previous = ledger.iloc[:-1]
    subsequent = ledger.iloc[1:]
    expected_values = (previous["portfolio"].to_numpy()
                       + previous["holdings"].to_numpy() * np.diff(S[0])
                       + subsequent["interest"].to_numpy()
                       - subsequent["fees"].to_numpy())
    record("daily self-financing portfolio identity", subsequent["portfolio"], expected_values)
    record("MSE decomposes into population variance plus squared bias",
           np.mean(err**2), np.var(err, ddof=0) + np.mean(err)**2)

    # Cost effects include foregone interest; an undiscounted fee sum cannot be
    # subtracted from a no-cost terminal error to recreate the costed hedge.
    nofee_err, _, _ = hedge_paths(S, target, 100, r, 0, 2, E)
    terminal_fee_burden = f * (50 * g**4 + 33 * g**2 + 24)
    record("cash account compounds each fee to expiry", nofee_err - err, [terminal_fee_burden])

    # A vectorized run must agree with independently processed selected paths.
    S = np.array([[100, 105, 90, 110], [100, 97, 104, 99]], dtype=float)
    target = np.array([[0.5, 0.7, 0.3, np.nan], [0.6, 0.4, 0.8, np.nan]])
    all_err, all_fees, _ = hedge_paths(S, target, 100, 0.04, 5, 1, 9)
    for i in range(2):
        one_err, one_fees, _ = hedge_paths(S[i:i + 1], target[i:i + 1], 100, 0.04, 5, 1, 9)
        record(f"path {i} vectorized errors equal individual accounting", all_err[i], one_err[0])
        record(f"path {i} vectorized fees equal individual accounting", all_fees[i], one_fees[0])
    record("nondegenerate loss decomposition", np.mean(all_err**2),
           np.var(all_err, ddof=0) + np.mean(all_err)**2)

    return {"status": "PASS", "check_count": len(checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_validation()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
