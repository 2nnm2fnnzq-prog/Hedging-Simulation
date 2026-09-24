"""European Heston calls and the three hedge inputs (zero dividends).

The Fourier calculation uses the stable characteristic function and a
Black--Scholes control variate. Spot and *variance* derivatives are analytic
derivatives of the integral. A 144/96-node comparison checks each point; on
disagreement, a 180-node refinement must agree with 144 at the same tolerances.
Unresolved points use QuantLib's adaptive integral and five-point differences.
Every refinement, fallback and boundary correction is counted.

QuantLib implementation reference:
https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/analytichestonengine.cpp
"""

from collections import Counter
from functools import lru_cache
import math

import numpy as np
from scipy.special import ndtr, roots_laguerre

_DIAGNOSTICS = Counter()
_SQRT_2PI = np.sqrt(2.0 * np.pi)
DEFAULT_ORDER = 144


def get_diagnostics(reset=False):
    result = dict(_DIAGNOSTICS)
    if reset:
        _DIAGNOSTICS.clear()
    return result


def _parameters(params):
    p = tuple(float(params[k]) for k in ("kappa", "theta", "xi", "rho"))
    kappa, theta, xi, rho = p
    if not (kappa > 0 and theta > 0 and xi > 0 and abs(rho) < 1):
        raise ValueError("Require kappa, theta, xi > 0 and abs(rho) < 1")
    return p


@lru_cache(maxsize=8192)
def _coefficients(tau, p, order):
    # Laguerre weights incorporate exp(-u); undo that weight in log space.
    u, laguerre_w = roots_laguerre(order)
    weights = np.exp(np.log(laguerre_w) + u) / (u * u + 0.25)
    z = u - 0.5j
    kappa, theta, xi, rho = p
    a = kappa - rho * xi * 1j * z
    d = np.sqrt(a * a + xi * xi * (z * z + 1j * z))
    # Rationalize a-d to preserve the constant-variance limit when xi is small.
    scaled_difference = -(z * z + 1j * z) / (a + d)
    g = xi * xi * scaled_difference / (a + d)
    ed = np.exp(-d * tau)
    c = kappa * theta / xi**2 * (
        xi * xi * scaled_difference * tau - 2 * np.log1p(g * (-np.expm1(-d*tau)) / (1-g))
    )
    dv = scaled_difference * (-np.expm1(-d * tau)) / (1 - g * ed)
    mean_weight = -np.expm1(-kappa * tau) / (kappa * tau)
    control_coefficient = -0.5 * tau * (z * z + 1j * z)
    return u, weights, c, dv, mean_weight, control_coefficient


def _fourier(S, v, K, tau, r, p, order):
    u, weights, c, dv, a, cb = _coefficients(tau, p, order)
    avg_v = p[1] + (v - p[1]) * a
    sigma = np.sqrt(avg_v)
    root_t = np.sqrt(tau)
    d1 = (np.log(S / K) + r * tau + 0.5 * avg_v * tau) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    bs = S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)
    bs_dv = S * np.exp(-0.5 * d1**2) / _SQRT_2PI * root_t * a / (2 * sigma)
    heston_cf = np.exp(c[None, :] + v[:, None] * dv[None, :])
    control_cf = np.exp(avg_v[:, None] * cb[None, :])
    phase = np.exp(1j * (np.log(S / K) + r * tau)[:, None] * u[None, :])
    integrand = phase * (heston_cf - control_cf)
    prefactor = np.sqrt(S * K) * np.exp(-0.5 * r * tau) / np.pi
    price = bs - prefactor * np.sum(integrand.real * weights, axis=1)
    delta = ndtr(d1) - prefactor / S * np.sum(
        (integrand * (0.5 + 1j * u)).real * weights, axis=1
    )
    variance_derivative = bs_dv - prefactor * np.sum(
        (phase * (heston_cf * dv - control_cf * cb * a)).real * weights, axis=1
    )
    return price, delta, variance_derivative


def quantlib_price(S, v, K, tau, r, params, tolerance=1e-11):
    """Independent adaptive QuantLib price at an exact fractional-year tau.

    Time is rescaled to a 365-day Actual365Fixed year: r'=r*tau,
    v'=v*tau, kappa'=kappa*tau, theta'=theta*tau, xi'=xi*tau.
    This avoids rounding 1/252-year intervals to calendar days. QuantLib's
    positive-parameter constraint requires a 1e-12 variance at v=0; the
    caller records this numerical boundary convention.
    """
    import QuantLib as ql

    p = _parameters(params)
    kappa, theta, xi, rho = p
    date = ql.Date(1, 1, 2020)
    ql.Settings.instance().evaluationDate = date
    dc = ql.Actual365Fixed()
    risk_free = ql.YieldTermStructureHandle(ql.FlatForward(date, r * tau, dc))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(date, 0.0, dc))
    quote = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    process = ql.HestonProcess(
        risk_free, dividend, quote, max(float(v), 1e-12) * tau,
        kappa * tau, theta * tau, xi * tau, rho,
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model, tolerance, 20000)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
        ql.EuropeanExercise(date + 365),
    )
    option.setPricingEngine(engine)
    try:
        return float(option.NPV())
    except RuntimeError as exc:
        # Relative tolerance alone may not converge in nearly deterministic
        # deep-ITM/OTM tails. QuantLib's optimal-control-variate contour is the
        # second adaptive method, with an additional 1e-12 absolute criterion.
        _DIAGNOSTICS["quantlib_integrator_recoveries"] += 1
        second = ql.AnalyticHestonEngine(
            model, ql.AnalyticHestonEngine.OptimalCV,
            ql.AnalyticHestonEngine_Integration.gaussLobatto(tolerance, 1e-12, 50000), 1e-12,
        )
        option.setPricingEngine(second)
        try:
            return float(option.NPV())
        except RuntimeError as second_exc:
            raise RuntimeError(f"Both QuantLib integrations failed at S={S!r}, v={v!r}, K={K!r}, tau={tau!r}, r={r!r}, params={params!r}: {exc}; {second_exc}") from second_exc


def _quantlib_greeks(S, v, K, tau, r, params):
    price = quantlib_price(S, v, K, tau, r, params)
    hs = S * 1e-5
    delta = (
        quantlib_price(S - 2 * hs, v, K, tau, r, params)
        - 8 * quantlib_price(S - hs, v, K, tau, r, params)
        + 8 * quantlib_price(S + hs, v, K, tau, r, params)
        - quantlib_price(S + 2 * hs, v, K, tau, r, params)
    ) / (12 * hs)
    hv = min(1e-5, max(2e-7, 0.01 * (v + params["kappa"] * params["theta"] * tau / 2)))
    if v >= 2 * hv:
        variance_derivative = (
            quantlib_price(S, v - 2 * hv, K, tau, r, params)
            - 8 * quantlib_price(S, v - hv, K, tau, r, params)
            + 8 * quantlib_price(S, v + hv, K, tau, r, params)
            - quantlib_price(S, v + 2 * hv, K, tau, r, params)
        ) / (12 * hv)
    else:
        variance_derivative = (
            -25 * price
            + 48 * quantlib_price(S, v + hv, K, tau, r, params)
            - 36 * quantlib_price(S, v + 2 * hv, K, tau, r, params)
            + 16 * quantlib_price(S, v + 3 * hv, K, tau, r, params)
            - 3 * quantlib_price(S, v + 4 * hv, K, tau, r, params)
        ) / (12 * hv)
    return price, delta, variance_derivative


def heston_price_greeks(S, v, K, tau, r, params, *, quadrature_order=DEFAULT_ORDER, check=True):
    """Return (call price, spot partial, variance partial), preserving array shape.

    Default quadrature agreement tolerances are 1e-7 dollars, 2e-6 shares,
    and 2e-4 dollars per variance unit. A failed 144/96 check first receives
    a 180-node refinement, tested against 144 with the same tolerances.
    Adaptive refinements and QuantLib fallbacks are counted.
    `check=False` is provided only for the explicit numerical-convergence audit.
    """
    S, v = np.broadcast_arrays(np.asarray(S, float), np.asarray(v, float))
    shape = S.shape
    s, vv = S.ravel(), v.ravel()
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(vv)) or np.any(s <= 0) or np.any(vv < 0):
        _DIAGNOSTICS["invalid_state_failures"] += 1
        raise ValueError("Heston state must have finite S > 0 and v >= 0")
    if not (np.isfinite(K) and K > 0 and np.isfinite(tau) and tau >= 0 and np.isfinite(r)):
        raise ValueError("Invalid contract or rate")
    p = _parameters(params)
    _DIAGNOSTICS["heston_points"] += len(s)
    if tau == 0:
        return tuple(a.reshape(shape) for a in (np.maximum(s - K, 0), (s > K).astype(float), np.zeros_like(s)))
    if not 24 <= quadrature_order <= 180:
        raise ValueError("quadrature_order must be between 24 and 180")
    result = list(_fourier(s, vv, K, tau, r, p, quadrature_order))
    lower = np.maximum(s - K * np.exp(-r * tau), 0)
    invalid = (
        ~np.isfinite(result[0]) | ~np.isfinite(result[1]) | ~np.isfinite(result[2])
        | (result[0] < lower - 1e-8) | (result[0] > s + 1e-8)
        | (result[1] < -1e-7) | (result[1] > 1 + 1e-7)
    )
    if check:
        low_order = max(24, int(quadrature_order * 2 / 3))
        comparison = _fourier(s, vv, K, tau, r, p, low_order)
        disagreement = (
            np.abs(result[0] - comparison[0]) > 1e-7
        ) | (np.abs(result[1] - comparison[1]) > 2e-6) | (
            np.abs(result[2] - comparison[2]) > 2e-4
        )
        invalid |= disagreement
        _DIAGNOSTICS["quadrature_disagreements"] += int(disagreement.sum())
        if quadrature_order == DEFAULT_ORDER and np.any(invalid):
            indices = np.flatnonzero(invalid)
            refined = _fourier(s[indices], vv[indices], K, tau, r, p, 180)
            converged = (
                (np.abs(result[0][indices] - refined[0]) <= 1e-7)
                & (np.abs(result[1][indices] - refined[1]) <= 2e-6)
                & (np.abs(result[2][indices] - refined[2]) <= 2e-4)
                & np.isfinite(refined[0]) & np.isfinite(refined[1]) & np.isfinite(refined[2])
                & (refined[0] >= lower[indices] - 1e-8) & (refined[0] <= s[indices] + 1e-8)
                & (refined[1] >= -1e-7) & (refined[1] <= 1 + 1e-7)
            )
            accepted_indices = indices[converged]
            _DIAGNOSTICS["quadrature_refinement_points"] += len(indices)
            _DIAGNOSTICS["quadrature_refinement_acceptances"] += int(converged.sum())
            for a, b in zip(result, refined):
                a[accepted_indices] = b[converged]
            invalid[accepted_indices] = False
    for j in np.flatnonzero(invalid):
        _DIAGNOSTICS["quantlib_fallback_points"] += 1
        if vv[j] < 1e-12:
            _DIAGNOSTICS["quantlib_zero_variance_regularizations"] += 1
        try:
            replacement = _quantlib_greeks(float(s[j]), float(vv[j]), K, tau, r, params)
            for a, b in zip(result, replacement):
                a[j] = b
        except Exception:
            _DIAGNOSTICS["quantlib_fallback_failures"] += 1
            raise
    # Roundoff at the exact no-arbitrage boundary is recorded, never hidden.
    tolerance = 2e-8
    if np.any(result[0] < lower - tolerance) or np.any(result[0] > s + tolerance):
        _DIAGNOSTICS["price_bound_failures"] += 1
        raise ArithmeticError("Heston price outside no-arbitrage bounds after fallback")
    corrected = (result[0] < lower) | (result[0] > s)
    _DIAGNOSTICS["roundoff_price_bound_adjustments"] += int(corrected.sum())
    result[0] = np.clip(result[0], lower, s)
    if not all(np.all(np.isfinite(a)) for a in result):
        _DIAGNOSTICS["nonfinite_output_failures"] += 1
        raise ArithmeticError("Nonfinite Heston output")
    return tuple(a.reshape(shape) for a in result)


def bs_price(S, K, tau, r, sigma):
    """Call price using OTM option values to reduce cancellation."""
    S, sigma = np.broadcast_arrays(np.asarray(S, float), np.asarray(sigma, float))
    discounted_strike = K * np.exp(-r * tau)
    intrinsic = np.maximum(S - discounted_strike, 0)
    w = sigma * np.sqrt(tau)
    safe_w = np.maximum(w, 1e-300)
    d1 = np.log(S / discounted_strike) / safe_w + 0.5 * safe_w
    d2 = d1 - safe_w
    otm = np.where(S <= discounted_strike,
                   S * ndtr(d1) - discounted_strike * ndtr(d2),
                   discounted_strike * ndtr(-d2) - S * ndtr(-d1))
    return intrinsic + np.where(w > 0, otm, 0.0)


def bs_iv_delta(price, S, K, tau, r):
    """Contract-specific IV and BS spot partial, with IV held fixed.

    Zero time value at machine precision has an unidentifiable IV; use the
    zero-volatility limit and count those observations. Otherwise 55 bisection
    steps in total standard deviation provide a deterministic inversion.
    """
    price, S = np.broadcast_arrays(np.asarray(price, float), np.asarray(S, float))
    if tau <= 0 or K <= 0 or np.any(S <= 0) or not np.all(np.isfinite(price + S)):
        raise ValueError("IV inversion requires tau,K,S positive and finite inputs")
    discounted_strike = K * np.exp(-r * tau)
    intrinsic = np.maximum(S - discounted_strike, 0)
    if np.any(price < intrinsic - 2e-8) or np.any(price >= S):
        _DIAGNOSTICS["iv_bound_failures"] += 1
        raise ValueError("Reference call prices outside invertible bounds")
    _DIAGNOSTICS["iv_points"] += price.size
    boundary = price - intrinsic <= 1e-12
    _DIAGNOSTICS["iv_zero_time_value_limits"] += int(boundary.sum())
    lower = np.zeros(price.shape)
    upper = np.full(price.shape, 12.0)
    for _ in range(55):
        middle = (lower + upper) / 2
        proposed = bs_price(S, K, tau, r, middle / np.sqrt(tau))
        lower = np.where(proposed < price, middle, lower)
        upper = np.where(proposed >= price, middle, upper)
    total_vol = (lower + upper) / 2
    iv = np.where(boundary, 0.0, total_vol / np.sqrt(tau))
    x = np.log(S / discounted_strike)
    delta = ndtr(x / np.maximum(total_vol, 1e-300) + total_vol / 2)
    delta = np.where(boundary, np.where(x > 0, 1.0, np.where(x < 0, 0.0, 0.5)), delta)
    repricing = bs_price(S, K, tau, r, iv)
    if np.max(np.abs(repricing - price), initial=0) > 3e-8:
        _DIAGNOSTICS["iv_repricing_failures"] += 1
        raise ArithmeticError("IV inversion did not reproduce the supplied price")
    return iv, delta
