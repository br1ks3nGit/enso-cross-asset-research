# Peak El Niño insurance strategy

This package simulates a fixed-parameter daily strategy for the twelve tickers
shown in the supplied screenshot:

`ROOT, PLMR, KNSL, MCY, LMND, CINF, PGR, AIZ, AIG, BAP, ALL, HIG`.

## Reproduce from cached data

From the repository root:

```bash
MPLCONFIGDIR=/tmp/insurance_peak_mpl .venv/bin/python insurance_peak_strategy/run_strategy.py
.venv/bin/python insurance_peak_strategy/test_strategy.py
```

The daily OHLCV inputs are the Massive responses retained by the preceding
insurance/ONI study. Saved short-interest, short-volume, ticker-reference and
historical quote responses are read locally; no downloader is required. See the root
[`RUNBOOK.md`](../RUNBOOK.md) for the external ONI input variable.

## Output

- `peak_el_nino_insurance_strategy.html`: self-contained report.
- `calculated/overall_strategy_summary.csv`: linked-period portfolio results.
- `calculated/event_strategy_summary.csv`: one result per episode and strategy.
- `calculated/ticker_strategy_results.csv`: company/event backtests.
- `calculated/daily_portfolio_returns.csv`: quant-ready daily portfolio returns.
- `calculated/trade_log.csv`: entries, exits and ATR-stop outcomes.
- `calculated/signal_quality.csv`: score IC and next-day directional accuracy.
- `calculated/ticker_event_tendencies.csv`: volatility, MA, NBBO and short metrics.
- `calculated/alternative_data_coverage.csv`: actual vendor coverage.

## Fixed assumptions

- Severe El Niño: ONI at or above +1.5°C.
- Publication convention: three-month delay from the centered/revised ONI value.
- ATR: ten trading sessions, Wilder smoothing; trailing stop at 2.5× ATR.
- Execution: next-session open; forced close at the end of each eligible window.
- Cost: 7.5 basis points per transaction side and 3% annualized short borrow.
- Tax: 20% of positive realized event gains, applied when each event position is closed.
- Passive basket execution: fixed equal-dollar holdings, with one entry and one exit cost per holding and the same realized-gain tax.
- `KIE` is used as an investable US P&C insurance-sector benchmark proxy; it is not a pure underwriting or claims index.
- Initial capital: USD 100,000; equal-weight maximum gross exposure of 100%.
- Prices are split-adjusted and exclude dividends.

Short-interest history begins in December 2017. Short-volume history begins on
6 February 2024. Massive reference data provide shares outstanding, not true
free float. Weekly order-book direction uses the final NBBO strictly before
16:00 New York time on the week's last trading session.
