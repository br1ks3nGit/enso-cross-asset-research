# September–December 2026 ENSO forecast

Open `ENSO_forecast_September_December_2026.html` for the full standalone report. It contains ONI predictions and all 200 state-month ENSO temperature contribution estimates with uncertainty, source links and validation. `forecast_summary.md` contains a concise text version and all 50 states. Figures are independently shareable PNG files.

This is a research forecast, not an official NOAA forecast. ONI is the ERSSTv6 three-month centered index: September=ASO, October=SON, November=OND, December=NDJ (including January 2027). State estimates are statistical ENSO-associated differences from a neutral counterfactual, not identified causal effects or total departures from temperature normals. September–October have weak/no demonstrated additional forecast skill. The predicted event lies beyond the calibration sample's historical seasonal maxima.

## Files

- `raw/`: downloaded CDS NetCDF, NOAA original files and catalogue metadata.
- `requests/`: exact CDS request payloads, without authentication information.
- `download_manifest.jsonl`: download records with timestamps, sources, request parameters and checksums. A few initial public files were downloaded with curl; all raw files are covered by `raw_sha256.json` and source references in the report.
- `calculated/oni_forecast.json`: final Pacific predictions, comparison models, tuning results and 32 held-out forecast/actual pairs from eight years.
- `calculated/state_enso_effects.json`: all 200 state estimates, intervals, coefficients and validation results.
- `calculated/temperature_validation.json`: state temperature historical evaluation records, with observed-ONI and forecast-ONI variants distinguished.
- `calculated/state_temperature.jsonl.gz`, `division_temperature.jsonl.gz`: normalized monthly observations, including geographic identifiers and Celsius units.
- `calculated/oni_v6.jsonl`, `oni_v5.jsonl`: separately versioned NOAA indices.
- `calculated/oni_draws.npy`: correlated Student-t ONI scenarios used in effect uncertainty estimates.
- `calculated/era5_diagnostics.json`: spatially averaged Pacific diagnostic values.
- `versions.json`: software versions used to run the analysis.

## Reproduce offline

Run from the parent research workspace, using its `.venv` or another environment with the packages in `versions.json`:

```sh
.venv/bin/python enso_forecast_2026/prepare.py
.venv/bin/python enso_forecast_2026/model.py
.venv/bin/python enso_forecast_2026/report.py
.venv/bin/python -m unittest discover -s enso_forecast_2026 -p 'test_forecast.py' -v
```

The full model run resets its random seed to 20260911. Run the complete model script for reproducible uncertainty results; importing a single function in a fresh process starts the random stream at a different point.

## Download again

Configure your personal CDS token according to https://cds.climate.copernicus.eu/how-to-api and accept the dataset licences in your account. No credentials are included in this archive. `CDSAPI_RC` can identify a protected config file. The downloader also checks the task's temporary config if it still exists; that temporary file is removed when delivery is complete.

```sh
.venv/bin/python enso_forecast_2026/download.py public
.venv/bin/python enso_forecast_2026/download.py forecast --centre ecmwf --system 51
.venv/bin/python enso_forecast_2026/download.py hindcast --centre ecmwf --system 51
.venv/bin/python enso_forecast_2026/download.py forecast --centre dwd --system 22
.venv/bin/python enso_forecast_2026/download.py hindcast --centre dwd --system 22
.venv/bin/python enso_forecast_2026/download.py forecast --centre cmcc --system 4
.venv/bin/python enso_forecast_2026/download.py hindcast --centre cmcc --system 4
.venv/bin/python enso_forecast_2026/download.py era5
```

Existing downloaded filenames are cached. Exact requests target the 11 September 2026 analysis, not a generic latest-date refresh. For another forecast date, use a separate directory and update the initialization, systems, observation cutoff, target periods, model-year filters and date-specific NOAA filenames together. Do not mix new observations with this vintage's forecasts.

Historical file names use `hindcast` for convenience, but the actual coverage mixes modern-system reforecasts and forecast-era runs according to each provider. Validation is retrospective and uses current-vintage ONI and temperature observations, not immutable historical issue-time archives. DWD 2024 was absent; all ONI comparisons use the shared 32-year sample and exclude that year explicitly.

Contains modified Copernicus Climate Change Service information (2026). Neither the European Commission nor ECMWF is responsible for subsequent use. NOAA and the individual seasonal model providers are credited in the report.
