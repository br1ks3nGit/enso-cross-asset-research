#!/usr/bin/env python3
"""Integrity checks for the peak-El-Nino strategy package."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent


def main() -> None:
    overall = pd.read_csv(ROOT / "calculated" / "overall_strategy_summary.csv")
    events = pd.read_csv(ROOT / "calculated" / "event_strategy_summary.csv")
    tickers = pd.read_csv(ROOT / "calculated" / "ticker_strategy_results.csv")
    quality = pd.read_csv(ROOT / "calculated" / "signal_quality.csv")
    coverage = pd.read_csv(ROOT / "calculated" / "alternative_data_coverage.csv")
    trades = pd.read_csv(ROOT / "calculated" / "trade_log.csv")

    assert len(overall) == 5 and overall["series"].is_unique
    assert len(events) == 5 * events["episode"].nunique()
    assert set(coverage["ticker"]) == {"ROOT", "PLMR", "KNSL", "MCY", "LMND", "CINF", "PGR", "AIZ", "AIG", "BAP", "ALL", "HIG"}
    assert set(coverage["sv_start"]) == {"2024-02-06"}
    assert not events["episode"].str.contains("2026").any()
    assert events["return"].dropna().gt(-1).all()
    assert tickers["exposure"].dropna().between(0, 1).all()
    assert quality["direction_hit_rate"].dropna().between(0, 1).all()
    assert not np.isinf(events.select_dtypes(include=[np.number]).to_numpy()).any()

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    assert trades["exit_date"].notna().all()
    assert (trades["exit_date"] >= trades["entry_date"]).all()

    # Every sampled quote used for book direction is strictly before 16:00 New York time.
    for path in (ROOT / "raw" / "quotes_weekly_nbbo").glob("*/*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = json.load(handle).get("results", [])
        for row in rows:
            stamp = pd.to_datetime(row["sip_timestamp"], unit="ns", utc=True).tz_convert("America/New_York")
            assert (stamp.hour, stamp.minute, stamp.second) < (16, 0, 0)

    text = (ROOT / "peak_el_nino_insurance_strategy.html").read_text()
    soup = BeautifulSoup(text, "html.parser")
    assert len(soup.find_all("img")) == 3 + events["episode"].nunique()
    assert len(soup.find_all("table")) == 6
    for phrase in ("Key findings", "Rules fixed before testing", "Alternative-data coverage", "Metric formulas"):
        assert phrase in text

    key = json.loads((ROOT.parent / "config.json").read_text())["api_key"]
    for path in [ROOT / "peak_el_nino_insurance_strategy.html", ROOT / "feature_download_manifest.csv", *ROOT.glob("calculated/*.csv")]:
        if key:
            assert key not in path.read_text(errors="ignore")
    print("All strategy integrity checks passed.")


if __name__ == "__main__":
    main()
