# ONI COP/XLE options adaptation

Open `ONI_COP_XLE_options_adaptation_report.html` for the detailed report.

The analysis freezes the two non-overlapping 40-day strong-but-cooling ONI signals from the underlying COP/XLE walk-forward test. It reads retained historical option contracts selected as of each signal date, requires exact entry and exit trade bars, and compares five defined-risk option implementations using a 10% debit budget.

Reproduce from the workspace root:

```sh
.venv/bin/python enso_options_adaptation/run_analysis.py
.venv/bin/python enso_options_adaptation/test_analysis.py
```

The downloader is not called by this reproduction path.

Detailed contract selections, component prices, trades, strategy summaries, tax sensitivity and hashes are under `calculated/`.
