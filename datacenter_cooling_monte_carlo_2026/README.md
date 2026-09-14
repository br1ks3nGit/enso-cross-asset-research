# Data-center cooling supplier Monte Carlo, Oct–Dec 2026

This folder contains a scenario-conditioned stock-price simulation for MOD, AAON,
VRT, NVT, JCI, TT and SPXC, plus an equal-dollar basket.

## Reproduce from cached data

From the repository root:

```bash
.venv/bin/python datacenter_cooling_monte_carlo_2026/run_analysis.py
.venv/bin/python datacenter_cooling_monte_carlo_2026/test_analysis.py
```

The analysis fits the regularized factor models, runs 10,000 paths in each of 27 ONI
× temperature × metals cells, writes CSV diagnostics and builds the HTML report.
Saved monthly bars are read locally. The downloader is retained as acquisition
provenance only and is not part of this offline workflow; see the root
[`RUNBOOK.md`](../RUNBOOK.md).

All historical factor estimation ends in July 2026, the last complete month common
to the supplied ONI/metals data. Starting stock prices are 11 Sep 2026 adjusted closes.

## Main output

- `datacenter_cooling_stock_monte_carlo_oct_dec_2026.html`
- `calculated/all_27_scenario_results.csv`
- `calculated/factor_model_diagnostics.csv`
- `calculated/central_monthly_price_fan.csv`
- `calculated/central_market_regimes.csv`
- `calculated/oni_scenarios.csv`, `temperature_scenarios.csv`, `metal_scenarios.csv`
- `calculated/provenance.json`
- `calculated/news_attention_pricing.csv`
- `calculated/enso_news_event_windows.csv`

`analyze_news_pricing.py` reproduces the accompanying April–September 2026
news-attention and ENSO announcement-window diagnostics from the saved news files.

The simulation is a conditional sensitivity analysis, not a calibrated probability
forecast or a trading backtest. Historical company intercepts are set to zero in the
forward paths to avoid mechanically extrapolating recent AI-infrastructure stock gains.
