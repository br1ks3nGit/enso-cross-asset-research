# ENSO 2026 cooling capex and supplier research

Open `ENSO_2026_cooling_capex_supplier_research.html` for the standalone report.

Key outputs in `calculated/`:

- `state_cooling_capex_screen.csv`: temperature, disclosed capacity, central capex conclusion, contingency screen and pipeline basis.
- `public_supplier_screen.csv`: public cooling OEM and installer ranking.
- `massive_price_diagnostics.csv`: adjusted daily-price return diagnostics through 11 September 2026.
- `capex_assumptions.csv`: explicit low, central and high screening assumptions.

Reproduce from cached data at the repository root:

```sh
.venv/bin/python enso_datacenter_cooling_capex_2026/run_analysis.py
.venv/bin/python enso_datacenter_cooling_capex_2026/test_analysis.py
```

Saved price files are used; the downloader is not part of the offline workflow.

The central ENSO-attributable new-equipment capex estimate is approximately zero. The modeled contingency screen is hypothetical and must not be interpreted as committed operator capex or supplier revenue.
