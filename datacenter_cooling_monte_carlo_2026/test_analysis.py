#!/usr/bin/env python3
"""Structural and arithmetic checks for the cooling-stock Monte Carlo artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
CALC = ROOT / "calculated"


def main() -> None:
    results = pd.read_csv(CALC / "all_27_scenario_results.csv")
    assert len(results) == 27 * 8
    assert results["scenario"].nunique() == 27
    assert set(results["ticker"]) == {"MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC", "Basket"}
    assert (results.groupby("scenario")["ticker"].nunique() == 8).all()
    assert (results["p05_return"] <= results["p25_return"]).all()
    assert (results["p25_return"] <= results["median_return"]).all()
    assert (results["median_return"] <= results["p75_return"]).all()
    assert (results["p75_return"] <= results["p95_return"]).all()
    assert (results[["p05_end_price", "median_end_price", "p95_end_price"]] > 0).all().all()
    assert results["prob_positive"].between(0, 1).all()
    assert results["prob_loss20"].between(0, 1).all()
    assert results["prob_gain20"].between(0, 1).all()

    diagnostics = pd.read_csv(CALC / "factor_model_diagnostics.csv")
    assert len(diagnostics) == 7
    assert diagnostics["n"].min() >= 70
    assert diagnostics["resid_vol"].gt(0).all()

    fan = pd.read_csv(CALC / "central_monthly_price_fan.csv")
    assert len(fan) == 8 * 3
    assert (fan["p05"] <= fan["p25"]).all()
    assert (fan["p25"] <= fan["median"]).all()
    assert (fan["median"] <= fan["p75"]).all()
    assert (fan["p75"] <= fan["p95"]).all()

    provenance = json.loads((CALC / "provenance.json").read_text())
    assert provenance["paths_per_scenario"] == 10_000
    assert provenance["scenario_count"] == 27
    assert len(provenance["inputs"]) >= 10

    report = ROOT / "datacenter_cooling_stock_monte_carlo_oct_dec_2026.html"
    soup = BeautifulSoup(report.read_text(), "html.parser")
    assert soup.title and "Monte Carlo" in soup.title.text
    assert len(soup.find_all("table")) >= 10
    assert len(soup.find_all("svg")) >= 1
    assert "all 216 company × scenario results" in soup.get_text(" ")
    assert "apiKey" not in report.read_text()
    print("All Monte Carlo artifact checks passed.")


if __name__ == "__main__":
    main()
