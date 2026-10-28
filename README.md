# End-of-Day Options Trading Bot (Python + ThetaData)

Buys a short-dated SPY call or put at the market close when a signal fires, and
sells it at a later close. Historical option quotes come from ThetaData. It
includes a full trade-by-trade backtester with real bid/ask spreads,
commissions, stop-loss and take-profit.

> Educational project. Not financial advice. Options can lose 100% of the premium.

## How it works

```
ThetaData (or synthetic) ─► underlying closes ─► signals.py ─► pick_contract() ─► backtest.py ─► report
                         └► option EOD bid/ask/close ─────────────────────────────┘
```

| Step | File | What it does |
|---|---|---|
| Getting options data | `optbot/data.py` | `ThetaDataSource` calls the local Theta Terminal REST API (stock EOD, expirations, strikes, option EOD) and caches every response to disk. `SyntheticSource` prices a realistic chain with Black-Scholes, a vol smile and widening spreads, so the bot runs offline. |
| End-of-day strategy | `optbot/data.py::pick_contract` | At the close: the nearest expiration 5–21 days out, with the strike closest to spot (ATM). |
| Trading signals | `optbot/signals.py::basic_signal` | If the day's move is beyond ±0.5%, fade it (mean reversion) or follow it (momentum). |
| Advanced signals | `optbot/signals.py::advanced_signal` | RSI(3) oversold or overbought, plus a 200-day SMA trend filter (buy dips in uptrends, sell rips in downtrends), plus an implied-vs-realised volatility filter that skips trades when options are overpriced. |
| Backtesting | `optbot/backtest.py` | Fills halfway between mid and ask/bid, charges $0.65/contract, risks 2% of equity per trade, checks stop/target at each close, holds one position at a time. |
| Report | `optbot/report.py` | Equity vs underlying, trade markers, per-trade return histogram, and CSVs of every trade. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**For real data**, you need a ThetaData account with an options subscription.
Download the Theta Terminal from thetadata.net and start it:

```bash
java -jar ThetaTerminal.jar YOUR_EMAIL YOUR_PASSWORD
```

It listens on `http://127.0.0.1:25510`. Change `thetadata.base_url` in
`config.yaml` if yours differs. If ThetaData has moved to a newer API version,
the four endpoint paths are all in `ThetaDataSource` in `optbot/data.py`.

## Usage

```bash
python main.py backtest --compare                     # basic vs advanced, real ThetaData
python main.py backtest --compare --source synthetic  # offline demo
python main.py backtest --signal basic                # one strategy
python main.py signal                                 # today's signal + contract it would buy
pytest -q                                             # tests (offline)
```

Results go to `reports/`: `options_backtest.png`, `metrics.csv`, `trades_<signal>.csv`.

## Tuning

All parameters are in `config.yaml`: the underlying, date range, DTE window,
moneyness, signal thresholds, hold period, sizing, stop/target and fill model.
Treat fills and spreads as the most important setting. On one-day option holds,
the spread is often bigger than the edge, so test with
`slippage_pct_of_spread: 1.0` (full ask/bid) before you trust a result.

## Limitations

- Backtest only. It does not place orders; `signal` tells you what it would buy.
  To automate execution, add a broker client that buys the printed contract
  shortly before the close.
- EOD data means exits are only checked at closes. Intraday stops would trigger
  earlier, and at worse prices.
- Synthetic-data results only prove the code works, not that the strategy makes money.
