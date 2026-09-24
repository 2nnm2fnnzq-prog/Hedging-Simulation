"""Self-financing stock-and-cash hedges for a short European call.

The initial option-market price is the common cash endowment. Each strategy pays
its own stock-trading fees from that cash account, including its opening trade
and terminal liquidation. Errors are terminal liquidated cash minus the call
payoff, so a positive error is a surplus and a negative error is a shortfall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_paths(
    S: np.ndarray,
    target: np.ndarray,
    K: float,
    rate: float,
    fee_bps: float,
    rebalance_days: int,
    initial_endowment: float,
    trading_days: int = 252,
    ledger_path: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Run identical accounting conventions on every supplied stock path.

    ``S`` and ``target`` have shape ``(n_paths, maturity_days + 1)``. Targets
    are computed from contemporaneous information; this function executes the
    day-zero target and subsequently only targets at multiples of
    ``rebalance_days`` strictly before expiry. The expiry target is ignored.
    Interest compounds continuously at ``rate`` over each 1/``trading_days``
    year interval. Borrowing and lending use the same rate; there are no
    dividends, funding spreads, option trades, or position constraints.

    Returns terminal errors, the undiscounted sum of actual trading charges on
    each path, and an optional daily ledger for one zero-based path index.
    The fees' financing effects are already included in terminal cash. Do not
    subtract the returned fee sums from errors a second time.

    In the ledger, ``cash_before`` precedes that day's interest and trade;
    ``cash_after`` follows them. On expiry, ``portfolio`` and ``cash_after``
    are the liquidated value before the payoff is deducted. ``payoff`` and
    ``terminal_error`` are populated only on expiry.
    """
    spot = np.asarray(S, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    if spot.ndim != 2 or targets.shape != spot.shape:
        raise ValueError("S and target must have identical two-dimensional shapes")
    if spot.shape[0] < 1 or spot.shape[1] < 2:
        raise ValueError("At least one path and one time interval are required")
    if not np.all(np.isfinite(spot)) or np.any(spot <= 0):
        raise ValueError("All stock prices must be finite and strictly positive")
    if not np.all(np.isfinite(targets[:, :-1])):
        raise ValueError("All pre-expiry hedge targets must be finite")
    if not np.isfinite(K) or K <= 0:
        raise ValueError("K must be finite and strictly positive")
    if not np.isfinite(rate) or not np.isfinite(initial_endowment):
        raise ValueError("rate and initial_endowment must be finite scalars")
    if not np.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("fee_bps must be finite and nonnegative")
    for name, value in (("rebalance_days", rebalance_days), ("trading_days", trading_days)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    n_paths, n_dates = spot.shape
    if ledger_path is not None:
        if (isinstance(ledger_path, (bool, np.bool_))
                or not isinstance(ledger_path, (int, np.integer))
                or not 0 <= ledger_path < n_paths):
            raise ValueError("ledger_path must be a valid zero-based path index")

    maturity_day = n_dates - 1
    charge_rate = float(fee_bps) / 10_000.0
    growth_minus_one = np.expm1(float(rate) / trading_days)
    holdings = np.zeros(n_paths, dtype=np.float64)
    cash = np.full(n_paths, float(initial_endowment), dtype=np.float64)
    total_fees = np.zeros(n_paths, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    for day in range(n_dates):
        if ledger_path is not None:
            cash_before_selected = float(cash[ledger_path])
        interest = cash * growth_minus_one if day else np.zeros(n_paths)
        cash += interest
        if day == maturity_day:
            trade = -holdings
            event = "liquidate"
        elif day == 0 or day % rebalance_days == 0:
            trade = targets[:, day] - holdings
            event = "open" if day == 0 else "rebalance"
        else:
            trade = np.zeros(n_paths)
            event = "hold"
        charges = np.abs(trade) * spot[:, day] * charge_rate
        cash -= trade * spot[:, day] + charges
        holdings += trade
        total_fees += charges

        if ledger_path is not None:
            i = int(ledger_path)
            payoff = max(spot[i, day] - K, 0.0) if day == maturity_day else np.nan
            rows.append({
                "day": day,
                "time_years": day / trading_days,
                "event": event,
                "spot": float(spot[i, day]),
                "target_holdings": float(targets[i, day]) if day < maturity_day else np.nan,
                "cash_before": cash_before_selected,
                "interest": float(interest[i]),
                "trade": float(trade[i]),
                "fees": float(charges[i]),
                "cumulative_fees": float(total_fees[i]),
                "holdings": float(holdings[i]),
                "cash_after": float(cash[i]),
                "portfolio": float(cash[i] + holdings[i] * spot[i, day]),
                "payoff": float(payoff),
                "terminal_error": float(cash[i] - payoff) if day == maturity_day else np.nan,
            })

    errors = cash - np.maximum(spot[:, -1] - K, 0.0)
    ledger = pd.DataFrame.from_records(rows) if ledger_path is not None else None
    return errors, total_fees, ledger
