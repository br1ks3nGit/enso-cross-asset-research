# Insurance stocks and severe El Niño

This package downloads split-adjusted daily stock aggregates from Massive and
tests monthly insurance-stock returns against the supplied ONI series.

## Reproduce from cached data

From the repository root:

```bash
MPLCONFIGDIR=/tmp/insurance_oni_mpl .venv/bin/python insurance_oni_event_study/run_analysis.py
.venv/bin/python insurance_oni_event_study/test_analysis.py
```

This workflow uses the retained raw data and makes no API request. The downloader is
kept only as acquisition provenance and is not required for reproduction. See the
root [`RUNBOOK.md`](../RUNBOOK.md) for the external input variable and full execution
order.

## Output

- `insurance_ONI_event_study.html`: self-contained report with embedded charts.
- `prepared/download_manifest.csv`: vendor coverage, row counts, hashes, and errors.
- `prepared/daily_split_adjusted_prices_long.csv.gz`: quant-ready daily close panel.
- `prepared/monthly_log_returns_wide.csv.gz`: adjacent-month log returns.
- `calculated/company_ONI_statistics.csv`: one row per supplied company.
- `calculated/severe_event_company_paths.csv`: company-by-episode price reactions.
- `calculated/portfolio_benchmark_statistics.csv`: equal-weight insurer basket, KIE, and SPY.

## Important scope limits

- Massive returned stock history from 10 September 2003 for many long-lived
  tickers even though 1 January 2001 was requested. Later listings start on
  their actual or vendor-covered dates.
- Prices are adjusted for splits but do not include dividends.
- Chubb uses ACE through 14 January 2016 and CB thereafter. Everest uses RE
  through 7 July 2023 and EG thereafter.
- The ONI input is a revised monthly series, not a vintage archive. The
  available-information correlation applies a conservative three-month shift.
- The supplied 2026-active universe creates survivorship bias.
