"""Black-Scholes pricing and implied volatility (used by the synthetic data source and the IV filter)."""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float, right: str) -> float | None:
    intrinsic = max(0.0, S - K) if right == "C" else max(0.0, K - S)
    if T <= 0 or price <= intrinsic + 1e-6:
        return None
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, right) - price, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return None
