# 2026 data-center infrastructure long-straddle backtest

This study tests monthly ATM long straddles on VRT, MOD, ETN, PWR and GEV from
April 2026 through the last available session on 11 September 2026. Market data
come from Massive. The portfolio starts with HKD 100,000.

Run from the workspace root:

```sh
.venv/bin/python dc_infrastructure_straddles_2026/run_analysis.py
.venv/bin/python dc_infrastructure_straddles_2026/test_analysis.py
```

These commands use the saved option bars, contract selections and FX series. The
downloader is not part of the offline reproduction path.

The September holding period is partial and is never annualized or extrapolated.
