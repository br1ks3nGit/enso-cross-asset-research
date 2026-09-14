# Shared research inputs

These files are small, immutable inputs shared by several research packages. They
are stored once here so a GitHub checkout can reproduce the offline analyses without
machine-specific paths.

## `enso_oni_metals/`

The original ENSO/metal correlation package supplied for the research. The central
panel is `aligned_monthly_ONI_metals_1992_2026-07.csv`; the remaining CSVs are the
original descriptive, lag, stationarity, HAC, rolling-window and subperiod results
used by the metals report.

## `insurance/`

`el_nino_insurance_stocks_massive.csv` is the supplied insurer/event universe and
its source metadata.

Do not silently replace these files with newer vintages. Add a dated input alongside
the current file, record its source and checksum, and run it as a new experiment.
The environment variables documented in [`../RUNBOOK.md`](../RUNBOOK.md) may be used
to point the code at an authorized external copy without changing source code.
