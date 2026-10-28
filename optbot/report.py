"""Charts and CSVs for a backtest run."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_report(results: dict, under: pd.Series, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.4, 1.6, 1.2]})

    ax = axes[0]
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity, label=f"{name} signal", lw=1.4)
    first = next(iter(results.values()))
    u = under.reindex(first.equity.index).ffill()
    ax.plot(u.index, u / u.iloc[0] * first.equity.iloc[0], color="black", ls="--", lw=1,
            label="underlying (same capital)")
    ax.set_title("End-of-day options strategy — equity")
    ax.set_ylabel("$")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    main = results.get("advanced") or first
    ax.plot(u.index, u, color="#555", lw=1)
    t = main.trades
    if len(t):
        for right, mk, col in (("C", "^", "#2e7d32"), ("P", "v", "#c62828")):
            sel = t[t.right == right]
            ax.scatter(sel.entry_date, sel.spot, marker=mk, color=col, s=28, zorder=3,
                       label=f"{'call' if right == 'C' else 'put'} entries")
    ax.set_ylabel("underlying")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if len(t):
        ax.hist(t.return_pct.clip(-1, 2), bins=40, color="#1565c0", alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("option return per trade")
    ax.set_ylabel("# trades")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png = out / "options_backtest.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    pd.DataFrame({n: r.metrics for n, r in results.items()}).to_csv(out / "metrics.csv")
    for n, r in results.items():
        r.trades.to_csv(out / f"trades_{n}.csv", index=False)
        r.equity.rename("equity").to_csv(out / f"equity_{n}.csv")
    return png
