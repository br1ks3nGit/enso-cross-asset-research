#!/usr/bin/env python3
"""Integrity and arithmetic checks for the risk-management artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
CALC = ROOT / "calculated"
RETURN_COLUMNS = [
    "basic", "inverse_vol", "vol_target", "trend", "drawdown",
    "tail_budget", "combined", "core_signal", "confirmed", "XLI",
]


def main() -> None:
    monthly = pd.read_csv(CALC / "risk_management_monthly.csv")
    assert len(monthly) == 63 and monthly["month"].is_unique
    assert monthly["month"].iloc[0] == "2021-06" and monthly["month"].iloc[-1] == "2026-08"
    assert monthly[RETURN_COLUMNS].notna().all().all()
    assert monthly["combined_exposure"].between(0, 1).all()
    assert monthly["vol_multiplier"].between(0, 1).all()
    assert monthly["es_multiplier"].between(0, 1).all()
    assert monthly["dd_multiplier"].isin([1.0, 0.5, 0.25]).all()
    np.testing.assert_allclose(
        monthly["core_signal"], monthly["core_signal_gross"] - 0.0015 * monthly["core_signal_turnover"],
        rtol=0, atol=1e-12,
    )

    positions = pd.read_csv(CALC / "risk_management_positions.csv")
    assert len(positions) == 63 * 9 * 8
    stocks = positions[positions["ticker"] != "XLI_HEDGE"]
    capped = stocks[stocks["variant"] != "confirmed"]
    # The legacy two-name signal is retained as a comparison and predates the new
    # 25% policy. Every newly constructed basket, including the 85/15 mix, is capped.
    assert capped["weight"].between(0, 0.2500001).all()
    assert positions["gross_exposure"].between(0, 1.0000001).all()
    hedges = positions[positions["ticker"] == "XLI_HEDGE"]
    assert (hedges["hedge_weight"] <= 0).all()
    assert (positions["turnover"] >= 0).all() and (positions["cost"] >= 0).all()

    summary = pd.read_csv(CALC / "risk_management_summary.csv")
    assert set(summary["variant"]) == set(RETURN_COLUMNS)
    assert summary["annual_vol"].ge(0).all()
    assert summary["max_drawdown"].le(0).all()
    assert summary["var95_monthly"].ge(0).all() and summary["es95_monthly"].ge(0).all()
    basic = summary.set_index("variant").loc["basic"]
    assert abs(basic["cagr_retention_vs_basic"] - 1) < 1e-12
    assert abs(basic["drawdown_reduction_vs_basic"]) < 1e-12
    assert summary["meets_portfolio_gate"].any()

    policy = json.loads((CALC / "risk_policy.json").read_text())
    assert policy["initial_capital_hkd"] == 100_000
    assert policy["annual_vol_target"] == 0.20
    assert policy["max_name_weight"] == 0.25
    assert policy["core_signal_mix"] == {"risk_managed_core": 0.85, "confirmed_enso_signal": 0.15}
    assert policy["last_complete_month_state"]["information_through"] == "2026-08"

    allocation = pd.read_csv(CALC / "risk_management_next_allocation.csv")
    assert set(allocation["ticker"]) == {"MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC", "XLI hedge"}
    assert allocation.loc[allocation["ticker"] != "XLI hedge", "recommended_drawdown_brake_weight"].max() <= 0.2500001
    assert allocation["recommended_drawdown_brake_weight"].sum() <= 1.0000001

    report_path = ROOT / "cooling_strategy_risk_management_report.html"
    source = report_path.read_text()
    soup = BeautifulSoup(source, "html.parser")
    assert soup.title and "Risk management" in soup.title.text
    assert len(soup.find_all("table")) >= 3
    assert len(soup.find_all("svg")) >= 3
    assert "API key" not in source and "apiKey" not in source
    assert "Maximum drawdown" in soup.get_text() and "Expected-shortfall budget" in soup.get_text()
    assert "README.md" in source
    print("All risk-management checks passed.")


if __name__ == "__main__":
    main()
