# ENSO / insurance / agricultural market research

Open `ENSO_insurance_agriculture_research.html` in a browser. Charts and the scenario selector work offline; all JavaScript is embedded.

The study downloaded historical prices using the configured Massive key. Futures probes returned HTTP 403. Public Yahoo futures quote histories are used only for descriptive price/volatility correlations; separately labelled Massive fund histories support the agriculture strategy simulations. The report documents this distinction and the limits of current-vintage NOAA climate data.

Main files:

- `ENSO_insurance_agriculture_research.html`: complete report, formulas, results, sources and interactive charts.
- `raw/`: original historical API responses and NOAA files. Failed futures-access probes are preserved.
- `calculated/results.json`: all aggregate results, model paths, probability forecasts and raw-file SHA-256 hashes.
- `calculated/daily_*.jsonl`: cleaned daily price and return series.
- `calculated/monthly_*.jsonl`: lagged climate joins and monthly outcomes.
- `calculated/weekly_enso.jsonl`, `calculated/oni.jsonl`: parsed climate series.

The study scripts and `test_enso_study.py` live in this folder. Python dependencies
are pandas, numpy, scipy, requests and plotly; tested with the versions recorded in
`versions.json`.

From that parent folder, regenerate entirely from cached data:

```sh
.venv/bin/python enso_study/enso_study_analyze.py
.venv/bin/python enso_study/enso_study_report.py
.venv/bin/python enso_study/test_enso_study.py
```

The downloader reads `api_key`, `start` and `end` from a separate local `config.json`. No API key or credential-containing configuration is included in the delivery archive. Do not overwrite an existing personal configuration. A new minimal configuration can have this form:

```json
{"api_key": "", "start": "2006-09-10", "end": "2026-09-09"}
```

Download stages are `probe`, `stocks`, `futures`, `public_futures`, `proxies`, and
`actions`, e.g. `.venv/bin/python enso_study/enso_study_download.py --stage stocks`.
They cache existing response filenames. Use a fresh copy/output folder for another
research window or a data refresh; mixing windows in the same cache is not supported.
Downloads need network access and a stock-data entitlement; cached analysis needs
neither. These scripts do not place trades.

The model is intentionally small: monthly conditional frequencies; Pearson/rank/controlled correlations; a fixed defensive exposure rule; chronological probability calibration; and calendar-year resampling for uncertainty. Full causal attribution, contract-level futures execution, catastrophe-exposure modelling and archived ENSO releases are not available in this dataset.
