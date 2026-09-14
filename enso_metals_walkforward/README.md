# ENSO, metals and cooling infrastructure walk-forward study

Open `ENSO_metals_cooling_walkforward_report.html` for the results.

The study uses the supplied revised monthly ONI/metal benchmark package and Massive monthly stock aggregates saved under `raw/`. The Massive futures catalog request returned HTTP 403 because the current key lacks futures entitlement, so the report keeps the contract-level futures backtest explicitly separate and unexecuted.

Reproduce the calculations from the workspace root:

```sh
python3 enso_metals_walkforward/run_analysis.py
python3 -m unittest discover -s enso_metals_walkforward -p 'test_*.py' -v
```

The analysis uses an expanding training sample, 12-month frozen test blocks, a two-month ONI publication delay, and one additional month between the information set and the target return. Generated CSV and JSON tables are in `calculated/`.

To complete the futures extension after enabling Massive futures access, query `/futures/v1/contracts` point-in-time for COMEX HG contracts from 2017-04-03 onward, download `/futures/v1/aggs/{ticker}` session or monthly bars, and construct each return within one raw contract. Do not bridge prices across expiries.
