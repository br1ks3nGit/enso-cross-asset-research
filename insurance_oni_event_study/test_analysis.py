#!/usr/bin/env python3
"""Focused integrity checks for the insurance/ONI research package."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    summary = pd.read_csv(ROOT / "calculated" / "company_ONI_statistics.csv")
    events = pd.read_csv(ROOT / "calculated" / "severe_event_company_paths.csv")
    portfolio = pd.read_csv(ROOT / "calculated" / "portfolio_benchmark_statistics.csv")
    manifest = pd.read_csv(ROOT / "prepared" / "download_manifest.csv")
    report = (ROOT / "insurance_ONI_event_study.html").read_text()

    assert len(summary) == 60 and summary["ticker"].is_unique
    assert int((summary["daily_rows"] > 0).sum()) == 58
    assert set(summary["quality"]) <= {"Usable", "Limited", "Insufficient"}
    assert int((manifest["status"] == "ok").sum()) == 62
    assert set(manifest.loc[manifest["status"] == "empty", "ticker"]) == {"ODMTY", "SNTAY"}
    assert summary.loc[summary["ticker"] == "CB", "first_date"].iat[0] == "2003-09-10"
    assert summary.loc[summary["ticker"] == "EG", "first_date"].iat[0] == "2003-09-10"
    assert len(events["episode"].unique()) == 4
    assert events["peak_oni"].max() == 2.6
    assert len(portfolio) == 3

    numeric = summary.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert summary["corr_available_r"].dropna().between(-1, 1).all()
    assert summary["hac_q_bh"].dropna().between(0, 1).all()
    assert summary["severe_q_bh"].dropna().between(0, 1).all()

    assert "data:image/png;base64," in report
    for phrase in ("Key findings", "Metric definitions", "All companies and data quality", "Insurance premiums are not losses"):
        assert phrase in report

    # The API key must never be written to artifacts.
    key = json.loads((ROOT.parent / "config.json").read_text())["api_key"]
    for path in [ROOT / "insurance_ONI_event_study.html", *ROOT.glob("prepared/*"), *ROOT.glob("calculated/*")]:
        if path.is_file():
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    text = handle.read()
            else:
                text = path.read_text(errors="ignore")
            if key:
                assert key not in text

    print("All integrity checks passed.")


if __name__ == "__main__":
    main()
