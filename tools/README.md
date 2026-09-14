# Repository utilities

- `build_readme_figures.py` rebuilds the four static charts used by the root README.
- `climate_join.py` performs a strict point-in-time join of local climate vintages to
  decision timestamps.
- `massive_history.py` and `query_massive_catalog.py` document the historical market
  data acquisition/archive workflow. They are not used by the offline research
  rebuild.
- `test_history.py` validates the archive planner and point-in-time join utility.

Run commands from the repository root. See [`../RUNBOOK.md`](../RUNBOOK.md) for exact
commands and the distinction between offline analysis and acquisition utilities.
