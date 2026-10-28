"""Offline tests — run with:  pytest -q"""
import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optbot.backtest import _fill, run_backtest
from optbot.config import load_config
from optbot.data import make_source, pick_contract
from optbot.pricing import bs_price, implied_vol
from optbot.signals import advanced_signal, basic_signal, rsi

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["data"]["source"] = "synthetic"
    return c


@pytest.fixture(scope="module")
def src(cfg):
    return make_source(cfg)


def test_put_call_parity():
    S, K, T, r, v = 400, 400, 30 / 365, 0.03, 0.2
    c, p = bs_price(S, K, T, r, v, "C"), bs_price(S, K, T, r, v, "P")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-8


def test_implied_vol_roundtrip():
    px = bs_price(400, 405, 14 / 365, 0.03, 0.27, "P")
    assert abs(implied_vol(px, 400, 405, 14 / 365, 0.03, "P") - 0.27) < 1e-4


def test_rsi_bounds():
    s = pd.Series(np.cumsum(np.random.default_rng(0).standard_normal(300)) + 100)
    r = rsi(s, 3)
    assert r.between(0, 100).all()


def test_basic_signal_directions(cfg):
    close = pd.Series([100, 101, 100, 100.1])          # +1%, -0.99%, +0.1%
    s = basic_signal(close, cfg)                        # mean reversion by default
    assert list(s) == [0, -1, 1, 0]


def test_signals_use_no_future_data(cfg, src):
    u = src.underlying_eod(date(2020, 1, 1), date(2023, 12, 29))["close"]
    full = advanced_signal(u, cfg)["signal"]
    cut = advanced_signal(u.iloc[:600], cfg)["signal"]
    assert (full.iloc[:600] == cut).all()


def test_contract_selection_respects_dte(cfg, src):
    d = date(2022, 3, 9)
    exp, k = pick_contract(src, cfg, d, 431.3, "C")
    assert cfg.contract.min_dte <= (exp - d).days <= cfg.contract.max_dte
    assert k == 431.0


def test_fill_crosses_half_spread(cfg):
    row = pd.Series({"bid": 1.00, "ask": 1.20, "close": 1.10})
    assert _fill(row, "buy", cfg) == pytest.approx(1.15)
    assert _fill(row, "sell", cfg) == pytest.approx(1.05)


def test_backtest_accounting(cfg, src):
    res = run_backtest(src, cfg)
    t = res.trades
    assert len(t) > 10
    assert (t.exit_date > t.entry_date).all()
    # equity after the last trade equals capital plus summed P&L
    assert t.equity.iloc[-1] == pytest.approx(cfg.risk.initial_capital + t.pnl.sum(), abs=0.05)
    # positions never overlap
    assert (t.entry_date.iloc[1:].values >= t.exit_date.iloc[:-1].values).all()
