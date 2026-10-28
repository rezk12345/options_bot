"""End-of-day options trading bot.

    python main.py backtest                       # strategy from config.yaml
    python main.py backtest --compare             # basic vs advanced side by side
    python main.py backtest --source synthetic    # offline, no ThetaData needed
    python main.py signal                         # today's signal + the contract it would buy
"""
from __future__ import annotations

import argparse
import copy
import logging
from datetime import date, timedelta

from optbot.backtest import atm_iv_series, run_backtest
from optbot.config import ROOT, load_config
from optbot.data import make_source, pick_contract
from optbot.report import write_report
from optbot.signals import make_signals


def fmt(v):
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)


def cmd_backtest(cfg, args):
    src = make_source(cfg)
    names = ["basic", "advanced"] if args.compare else [cfg.strategy.signal]
    results = {}
    for n in names:
        c = copy.deepcopy(cfg)
        c["strategy"]["signal"] = n
        print(f"\n=== backtest: {n} signal | {cfg.data.root} {cfg.data.start} -> {cfg.data.end} ===")
        res = run_backtest(src, c)
        results[n] = res
        for k, v in res.metrics.items():
            print(f"  {k:<20} {fmt(v)}")
        if len(res.trades):
            print("  exits:", res.trades.exit_reason.value_counts().to_dict())
    under = src.underlying_eod(date.fromisoformat(cfg.data.start), date.fromisoformat(cfg.data.end))["close"]
    png = write_report(results, under, ROOT / cfg.report.out_dir)
    print(f"\nreport -> {png}")


def cmd_signal(cfg, args):
    """Run the signal on the latest close and show the trade it implies (no orders are sent)."""
    src = make_source(cfg)
    end = date.fromisoformat(cfg.data.end) if cfg.data.source == "synthetic" else date.today()
    under = src.underlying_eod(end - timedelta(days=400), end)["close"]
    iv = atm_iv_series(src, cfg, under.tail(1)) if cfg.strategy.signal == "advanced" else None
    sig = make_signals(under, cfg, iv)
    last = sig.iloc[-1]
    d, spot = sig.index[-1], float(under.iloc[-1])
    print(f"{cfg.data.root} close {d}: {spot:.2f}")
    print(sig.tail(1).drop(columns=["close"], errors="ignore").T.to_string(header=False))
    s = int(last["signal"])
    if s == 0:
        print("\n-> no trade today")
        return
    right = "C" if s > 0 else "P"
    pc = pick_contract(src, cfg, d, spot, right)
    if not pc:
        print("\n-> signal fired but no expiration in the DTE window")
        return
    exp, k = pc
    q = src.option_eod(exp, k, right, d, d)
    quote = q.loc[d].to_dict() if d in q.index else {}
    print(f"\n-> BUY {cfg.data.root} {exp} {k:g}{right} at the close  quote={quote}"
          f"  | exit after {cfg.strategy.hold_days} day(s), stop -{cfg.risk.stop_loss_pct:.0%},"
          f" target +{cfg.risk.take_profit_pct:.0%}")


def main():
    p = argparse.ArgumentParser(description="End-of-day options trading bot")
    p.add_argument("command", choices=["backtest", "signal"])
    p.add_argument("--config")
    p.add_argument("--source", help="thetadata | synthetic")
    p.add_argument("--signal", help="basic | advanced")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.source:
        cfg["data"]["source"] = a.source
    if a.signal:
        cfg["strategy"]["signal"] = a.signal
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"backtest": cmd_backtest, "signal": cmd_signal}[a.command](cfg, a)


if __name__ == "__main__":
    main()
