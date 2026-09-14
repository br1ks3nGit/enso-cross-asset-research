# Strong El Niño COP/XLE walk-forward simulation

Open `COP_XLE_strong_El_Nino_walkforward_report.html` for the detailed report.

The analysis reads saved daily split-adjusted COP and XLE bars plus split-adjusted dividend records. It applies a two-month embargo to the supplied revised ONI series, learns direction only from completed earlier signals, and evaluates five positive strong-El-Niño rules over 1, 2, 5, 10, 20, 40 and 60 trading days.

Reproduce from the workspace root:

```sh
.venv/bin/python enso_cop_xle_strong_event_walkforward/run_analysis.py
.venv/bin/python enso_cop_xle_strong_event_walkforward/test_analysis.py
```

This reproduction path makes no API request. See the root
[`RUNBOOK.md`](../RUNBOOK.md) for the shared ONI input.

Detailed observations, non-overlapping trades, strategy summaries, signal-strength tables, regime tables, tax sensitivity and hashes are under `calculated/`.
