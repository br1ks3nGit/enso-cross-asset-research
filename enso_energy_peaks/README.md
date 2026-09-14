# El Niño peaks and U.S. energy equities

Open `El_Nino_peaks_US_energy_backtest.html` for the report.

The study uses Massive monthly split-adjusted bars and Massive dividend adjustment factors for XOM, CVX, COP, EOG, OXY, SLB, MPC, PSX, KMI, WMB, XLE and SPY. It identifies ONI-defined warm episodes, tests returns after their retrospective peaks, estimates delayed-ONI predictive correlations, and runs a six-month peak-confirmation strategy whose direction uses only completed prior events.

Reproduce from the workspace root:

```sh
python3 enso_energy_peaks/run_analysis.py
python3 -m unittest discover -s enso_energy_peaks -p 'test_*.py' -v
```

All detailed event, correlation and strategy tables are under `calculated/`. API credentials are not stored in the report or provenance file.
