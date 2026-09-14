# ENSO 2026 data-center cooling-spend estimate

This analysis converts the existing state ENSO temperature contribution into a
December 2026 data-center cooling-electricity scenario for states above +2°C.

Run from the workspace root:

```sh
python3 enso_datacenter_cooling_2026/run_analysis.py
python3 -m unittest discover -s enso_datacenter_cooling_2026 -p 'test_*.py' -v
```

Absolute state totals are calculated only where the cited capacity paper reports
the state separately. All states also receive a normalized estimate per 100 MW of
total facility power capacity.
