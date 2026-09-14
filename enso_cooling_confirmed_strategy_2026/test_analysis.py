#!/usr/bin/env python3
"""Integrity checks for the confirmed ENSO cooling strategy artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
CALC = ROOT / "calculated"


def main() -> None:
    bt = pd.read_csv(CALC / "backtest_monthly.csv")
    assert len(bt) == 63 and bt["month"].is_unique
    assert bt["month"].iloc[0] == "2021-06" and bt["month"].iloc[-1] == "2026-08"
    assert bt[["strategy", "naive_oni_basket", "cooling_basket", "XLI"]].notna().all().all()
    assert bt["active"].sum() >= 5
    assert (bt["gross_exposure"] >= 0).all() and (bt["gross_exposure"] <= 1).all()

    positions = pd.read_csv(CALC / "backtest_positions.csv")
    assert len(positions) == len(bt) * 7
    assert positions.groupby("month")["selected"].sum().le(2).all()
    assert positions.groupby("month")["weight"].sum().le(1.0000001).all()

    perf = pd.read_csv(CALC / "backtest_summary.csv")
    assert len(perf) == 4
    assert perf["annual_vol"].ge(0).all()
    assert perf["max_drawdown"].le(0).all()

    forecast = pd.read_csv(CALC / "forecast_scenarios.csv")
    assert len(forecast) == 3 * 7
    assert set(forecast["scenario"]) == {"Excellent", "Normal", "Conservative"}
    assert set(forecast["ticker"]) == {"MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC"}
    assert (forecast["p10_end_price"] <= forecast["median_end_price"]).all()
    assert (forecast["median_end_price"] <= forecast["p90_end_price"]).all()
    assert forecast["prob_positive"].between(0, 1).all()
    assert (forecast.loc[forecast["scenario"] == "Excellent", "median_return"] > 0).all()

    portfolio = pd.read_csv(CALC / "forecast_strategy.csv")
    assert len(portfolio) == 3
    conservative = portfolio[portfolio["scenario"] == "Conservative"].iloc[0]
    assert conservative["holdings_oct"] == "Cash" and conservative["median_return"] == 0
    assert portfolio["prob_positive"].between(0, 1).all()

    provenance = json.loads((CALC / "provenance.json").read_text())
    assert provenance["paths_per_scenario"] == 20_000
    assert provenance["transaction_cost"] == 0.0015
    assert len(provenance["inputs"]) >= 19

    report = ROOT / "confirmed_enso_cooling_strategy_report.html"
    source = report.read_text()
    soup = BeautifulSoup(source, "html.parser")
    assert soup.title and "Confirmed ENSO" in soup.title.text
    assert len(soup.find_all("table")) >= 6
    assert len(soup.find_all("svg")) == 1
    assert "apiKey" not in source
    assert "Excellent" in soup.get_text() and "Conservative" in soup.get_text()
    print("All confirmed-strategy checks passed.")


if __name__ == "__main__":
    main()
