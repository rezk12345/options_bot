"""End-of-day trading signals.

Every signal is computed from data available at today's close only and returns
a Series indexed by date with values:
    +1 -> buy a call at today's close
    -1 -> buy a put  at today's close
     0 -> no trade
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def realised_vol(close: pd.Series, window: int) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)


def basic_signal(close: pd.Series, cfg: Config) -> pd.Series:
    """Move over `lookback` days beyond `threshold` -> trade it (fade or follow)."""
    b = cfg.strategy.basic
    move = close.pct_change(b.lookback)
    sig = pd.Series(0, index=close.index)
    direction = -1 if b.mode == "mean_reversion" else 1
    sig[move >= b.threshold] = direction
    sig[move <= -b.threshold] = -direction
    return sig.astype(int)


def advanced_signal(close: pd.Series, cfg: Config, atm_iv: pd.Series | None = None) -> pd.DataFrame:
    """RSI extremes, confirmed by the trend and filtered on option richness.

    - Oversold RSI + a down day  -> call candidate (bounce)
    - Overbought RSI + an up day -> put candidate (pullback)
    - Trend filter: calls only above the SMA, puts only below it
      (buy dips in uptrends, sell rips in downtrends). Use a SHORT RSI (2-3)
      with a LONG SMA (~200): a 14-day RSI is rarely oversold while price is
      still above a 50-day SMA, so that combination almost never fires.
    - Vol filter: skip when implied vol is far above realised vol, because
      overpriced premium eats the edge of a one-day hold.

    Returns a DataFrame with the final 'signal' plus every component, so you
    can see *why* each trade fired.
    """
    a = cfg.strategy.advanced
    df = pd.DataFrame({"close": close})
    df["ret"] = close.pct_change()
    df["rsi"] = rsi(close, a.rsi_period)
    df["sma"] = close.rolling(a.trend_sma).mean() if a.trend_sma else np.nan
    df["rv"] = realised_vol(close, a.vol_window)
    df["iv"] = atm_iv.reindex(df.index) if atm_iv is not None else np.nan

    call = (df["rsi"] < a.rsi_oversold) & (df["ret"] <= -a.min_move)
    put = (df["rsi"] > a.rsi_overbought) & (df["ret"] >= a.min_move)
    if a.trend_sma:
        call &= df["close"] > df["sma"]
        put &= df["close"] < df["sma"]

    df["iv_premium"] = df["iv"] / df["rv"]
    rich = df["iv_premium"] > a.max_iv_premium     # NaN compares False -> filter off without IV
    df["filtered_rich"] = (call | put) & rich
    call &= ~rich
    put &= ~rich

    df["signal"] = 0
    df.loc[call, "signal"] = 1
    df.loc[put, "signal"] = -1
    return df


def make_signals(close: pd.Series, cfg: Config, atm_iv: pd.Series | None = None) -> pd.DataFrame:
    if cfg.strategy.signal == "basic":
        return pd.DataFrame({"close": close, "signal": basic_signal(close, cfg)})
    return advanced_signal(close, cfg, atm_iv)
