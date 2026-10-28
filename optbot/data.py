"""Options data sources.

Both sources expose the same interface, so the strategy and backtester never
know where the data came from:

    underlying_eod(start, end)            -> DataFrame[date -> close]
    expirations()                         -> sorted list[date]
    option_eod(exp, strike, right, s, e)  -> DataFrame[date -> bid, ask, close]

ThetaDataSource talks to a locally running Theta Terminal over its REST API.
SyntheticSource prices a realistic chain with Black-Scholes on a simulated
underlying with stochastic volatility, so the full pipeline runs offline.
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config
from .pricing import bs_price

log = logging.getLogger(__name__)
RISK_FREE = 0.03


def _ymd(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def _from_ymd(x) -> date:
    return datetime.strptime(str(int(x)), "%Y%m%d").date()


class OptionsDataSource(ABC):
    root: str

    @abstractmethod
    def underlying_eod(self, start: date, end: date) -> pd.DataFrame: ...
    @abstractmethod
    def expirations(self) -> list[date]: ...
    @abstractmethod
    def option_eod(self, exp: date, strike: float, right: str, start: date, end: date) -> pd.DataFrame: ...


# =================================================================== ThetaData
class ThetaDataSource(OptionsDataSource):
    """Client for the Theta Terminal v2 REST API (default http://127.0.0.1:25510).

    Strikes are sent in tenths of a cent (450.0 -> 450000) and dates as YYYYMMDD,
    per ThetaData's v2 conventions. Responses are cached to disk as JSON so a
    backtest only downloads each contract once.
    """

    def __init__(self, cfg: Config):
        import requests

        self.http = requests.Session()
        self.base = cfg.thetadata.base_url.rstrip("/")
        self.timeout = cfg.thetadata.timeout
        self.root = cfg.data.root
        self.cache = ROOT / cfg.data.cache_dir / "thetadata"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, **params) -> pd.DataFrame:
        key = path.strip("/").replace("/", "_") + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        f = self.cache / f"{key}.json"
        if f.exists():
            payload = json.loads(f.read_text())
        else:
            try:
                r = self.http.get(self.base + path, params=params, timeout=self.timeout)
            except Exception as e:
                raise RuntimeError(f"cannot reach Theta Terminal at {self.base} — is it running? ({e})") from e
            if r.status_code == 472:      # ThetaData: no data for this request
                return pd.DataFrame()
            r.raise_for_status()
            payload = r.json()
            f.write_text(json.dumps(payload))
        fmt = payload.get("header", {}).get("format")
        rows = payload.get("response", [])
        if fmt:
            return pd.DataFrame(rows, columns=fmt)
        return pd.DataFrame({"value": rows})

    def underlying_eod(self, start, end) -> pd.DataFrame:
        df = self._get("/v2/hist/stock/eod", root=self.root, start_date=_ymd(start), end_date=_ymd(end))
        if df.empty:
            raise RuntimeError("no underlying data returned")
        df["date"] = df["date"].map(_from_ymd)
        return df.set_index("date")[["close"]].astype(float)

    def expirations(self) -> list[date]:
        df = self._get("/v2/list/expirations", root=self.root)
        return sorted(_from_ymd(x) for x in df.iloc[:, 0])

    def strikes(self, exp: date) -> list[float]:
        df = self._get("/v2/list/strikes", root=self.root, exp=_ymd(exp))
        return sorted(float(x) / 1000 for x in df.iloc[:, 0])

    def option_eod(self, exp, strike, right, start, end) -> pd.DataFrame:
        df = self._get("/v2/hist/option/eod", root=self.root, exp=_ymd(exp),
                       strike=int(round(strike * 1000)), right=right,
                       start_date=_ymd(start), end_date=_ymd(end))
        if df.empty:
            return df
        df["date"] = df["date"].map(_from_ymd)
        return df.set_index("date")[["bid", "ask", "close"]].astype(float)


# =================================================================== Synthetic
class SyntheticSource(OptionsDataSource):
    """Simulated SPY-like underlying + Black-Scholes option chain with a vol smile.

    - Underlying follows GBM with mean-reverting stochastic volatility.
    - Implied vol = realised regime vol * a variance-risk premium (options are
      usually a bit rich), plus a skew so OTM puts cost more.
    - Bid/ask spread widens for cheap and far-OTM contracts, like real markets.
    """

    def __init__(self, cfg: Config, seed: int = 11):
        self.root = cfg.data.root
        rng = np.random.default_rng(seed)
        start = pd.Timestamp(cfg.data.start) - pd.Timedelta(days=400)   # warm-up for indicators
        days = pd.bdate_range(start, pd.Timestamp(cfg.data.end) + pd.Timedelta(days=60))
        n = len(days)
        vol = np.empty(n)
        vol[0] = 0.16
        for t in range(1, n):   # mean-reverting log-vol with shocks clustered in sell-offs
            vol[t] = math.exp(math.log(vol[t - 1]) + 0.04 * (math.log(0.16) - math.log(vol[t - 1]))
                              + 0.09 * rng.standard_normal())
        vol = np.clip(vol, 0.08, 0.80)
        z = rng.standard_normal(n)
        # slight short-term mean reversion so EOD signals have something to find
        rets = np.empty(n)
        rets[0] = 0
        for t in range(1, n):
            rets[t] = 0.0003 - 0.06 * rets[t - 1] + vol[t] / math.sqrt(252) * z[t]
        px = 380 * np.exp(np.cumsum(rets))
        self.px = pd.Series(px, index=[d.date() for d in days])
        self.vol = pd.Series(vol, index=self.px.index)
        self._exps = sorted({d.date() for d in days if d.weekday() in (0, 2, 4)})  # Mon/Wed/Fri expiries

    def underlying_eod(self, start, end) -> pd.DataFrame:
        s = self.px.loc[start:end]
        return s.to_frame("close")

    def expirations(self) -> list[date]:
        return self._exps

    def iv(self, d: date, K: float) -> float:
        S = self.px[d]
        m = math.log(K / S)
        skew = -0.8 * m + 2.0 * m * m             # puts (K<S) richer
        return max(0.05, self.vol[d] * 1.12 + skew * self.vol[d])

    def option_eod(self, exp, strike, right, start, end) -> pd.DataFrame:
        rows = {}
        for d in self.px.loc[start:min(end, exp)].index:
            T = max((exp - d).days, 0) / 365
            mid = bs_price(self.px[d], strike, T, RISK_FREE, self.iv(d, strike), right)
            spread = max(0.01, min(0.10 * mid, 0.02 + 0.015 * mid))
            bid = max(0.0, mid - spread / 2)
            rows[d] = {"bid": round(bid, 2), "ask": round(mid + spread / 2, 2), "close": round(mid, 2)}
        return pd.DataFrame.from_dict(rows, orient="index")


def make_source(cfg: Config) -> OptionsDataSource:
    if cfg.data.source == "synthetic":
        return SyntheticSource(cfg)
    if cfg.data.source == "thetadata":
        return ThetaDataSource(cfg)
    raise ValueError(f"unknown data source {cfg.data.source}")


# ============================================================ contract choice
def pick_contract(src: OptionsDataSource, cfg: Config, d: date, spot: float, right: str):
    """Nearest expiration within [min_dte, max_dte]; strike nearest to the moneyness target."""
    c = cfg.contract
    exps = [e for e in src.expirations() if c.min_dte <= (e - d).days <= c.max_dte]
    if not exps:
        return None
    exp = exps[0]
    sign = 1 if right == "C" else -1
    target = spot * (1 + sign * c.moneyness)
    if hasattr(src, "strikes"):
        ks = src.strikes(exp)
        strike = min(ks, key=lambda k: abs(k - target)) if ks else None
    else:
        strike = round(target / c.strike_step) * c.strike_step
    return None if strike is None else (exp, float(strike))
