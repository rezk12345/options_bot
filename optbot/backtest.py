"""Trade-by-trade options backtester for the end-of-day strategy.

For each signal day:
  1. pick a contract at today's close (nearest expiry in the DTE window, ATM-ish strike)
  2. buy at the ask side (mid + slippage share of the spread), pay commission
  3. check the option at each following close: stop-loss / take-profit / time exit
  4. sell at the bid side on exit
Only one position is open at a time; equity compounds trade by trade.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .data import RISK_FREE, OptionsDataSource, pick_contract
from .pricing import implied_vol
from .signals import make_signals

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    signals: pd.DataFrame
    metrics: dict


def _fill(row: pd.Series, side: str, cfg: Config) -> float:
    if cfg.execution.fill == "close" or not (row.get("bid", 0) > 0 and row.get("ask", 0) > 0):
        return float(row["close"])
    mid = (row["bid"] + row["ask"]) / 2
    # fraction of the half-spread paid beyond mid: 0 = fill at mid, 1 = full ask/bid
    half = (row["ask"] - row["bid"]) / 2 * cfg.execution.slippage_pct_of_spread
    return float(mid + half if side == "buy" else max(0.0, mid - half))


def atm_iv_series(src: OptionsDataSource, cfg: Config, close: pd.Series) -> pd.Series:
    """ATM call implied vol at each close (used by the advanced signal's richness filter)."""
    out = {}
    for d, spot in close.items():
        pc = pick_contract(src, cfg, d, spot, "C")
        if not pc:
            continue
        exp, k = pc
        q = src.option_eod(exp, k, "C", d, d)
        if q.empty or d not in q.index:
            continue
        px = float(q.loc[d, "close"])
        iv = implied_vol(px, spot, k, (exp - d).days / 365, RISK_FREE, "C")
        if iv:
            out[d] = iv
    return pd.Series(out, dtype=float)


def run_backtest(src: OptionsDataSource, cfg: Config) -> BacktestResult:
    start, end = date.fromisoformat(cfg.data.start), date.fromisoformat(cfg.data.end)
    warm = start - timedelta(days=int(cfg.strategy.advanced.trend_sma * 1.6) + 30)
    under = src.underlying_eod(warm, end)["close"]
    under.index = pd.Index(under.index)

    iv = None
    if cfg.strategy.signal == "advanced":
        log.info("computing ATM implied vol series...")
        iv = atm_iv_series(src, cfg, under.loc[start:])
    sig = make_signals(under, cfg, iv).loc[start:]
    days = list(under.loc[start:].index)
    pos_of = {d: i for i, d in enumerate(days)}

    r, ex = cfg.risk, cfg.execution
    equity = float(r.initial_capital)
    eq_curve = {days[0]: equity}
    trades = []
    busy_until = None

    for d in days:
        s = int(sig.at[d, "signal"]) if d in sig.index else 0
        if s == 0 or (busy_until and d < busy_until):
            eq_curve[d] = equity
            continue
        right = "C" if s > 0 else "P"
        spot = float(under[d])
        pc = pick_contract(src, cfg, d, spot, right)
        if not pc:
            continue
        exp, strike = pc
        i = pos_of[d]
        horizon = days[i:i + cfg.strategy.hold_days + 1]
        quotes = src.option_eod(exp, strike, right, d, horizon[-1])
        if quotes.empty or d not in quotes.index:
            continue

        entry = _fill(quotes.loc[d], "buy", cfg)
        if entry < 0.05:
            continue
        cost_per = entry * ex.contract_multiplier + ex.commission_per_contract
        n = min(r.max_contracts, int(equity * r.risk_per_trade // cost_per))
        if n < 1:
            continue

        exit_day, exit_px, reason = None, None, "time"
        for h in horizon[1:]:
            if h not in quotes.index:
                continue
            mark = float(quotes.loc[h, "close"])
            exit_day = h
            exit_px = _fill(quotes.loc[h], "sell", cfg)
            if mark <= entry * (1 - r.stop_loss_pct):
                reason = "stop"
                break
            if mark >= entry * (1 + r.take_profit_pct):
                reason = "target"
                break
            if h >= exp:
                reason = "expiry"
                break
        if exit_day is None:
            continue

        pnl = (exit_px - entry) * ex.contract_multiplier * n - 2 * ex.commission_per_contract * n
        equity += pnl
        busy_until = exit_day
        trades.append({"entry_date": d, "exit_date": exit_day, "right": right, "strike": strike,
                       "expiry": exp, "dte": (exp - d).days, "spot": round(spot, 2),
                       "contracts": n, "entry": round(entry, 2), "exit": round(exit_px, 2),
                       "pnl": round(pnl, 2), "return_pct": round(exit_px / entry - 1, 4),
                       "exit_reason": reason, "equity": round(equity, 2)})
        eq_curve[exit_day] = equity

    eq = pd.Series(eq_curve).sort_index()
    eq = eq.reindex(days).ffill()
    tdf = pd.DataFrame(trades)
    return BacktestResult(trades=tdf, equity=eq, signals=sig, metrics=compute_metrics(tdf, eq, under.loc[start:]))


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, under: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    yrs = max(len(equity) / 252, 1e-9)
    dd = (equity / equity.cummax() - 1).min()
    m = {"start_equity": equity.iloc[0], "end_equity": equity.iloc[-1],
         "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
         "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1,
         "sharpe": (r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else float("nan"),
         "max_drawdown": dd,
         "underlying_return": under.iloc[-1] / under.iloc[0] - 1,
         "trades": len(trades)}
    if len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        m.update(win_rate=len(wins) / len(trades),
                 avg_win=wins.pnl.mean() if len(wins) else 0.0,
                 avg_loss=losses.pnl.mean() if len(losses) else 0.0,
                 profit_factor=(wins.pnl.sum() / -losses.pnl.sum()) if losses.pnl.sum() < 0 else float("inf"),
                 calls=int((trades.right == "C").sum()), puts=int((trades.right == "P").sum()),
                 avg_option_return=trades.return_pct.mean())
    return m
