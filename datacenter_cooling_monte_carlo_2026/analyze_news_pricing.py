#!/usr/bin/env python3
"""News-attention and event-window diagnostics for ENSO pricing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NEWS = ROOT / "raw/news"
DAILY = ROOT.parent / "enso_datacenter_cooling_capex_2026/raw"
OUT = ROOT / "calculated"
TICKERS = ["MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC"]
ENSO = re.compile(r"\b(el ni(?:n|ñ)o|enso|oceanic niño|nino.?3|roni)\b", re.I)
THEME = re.compile(r"(data.?center|liquid cool|cooling|artificial intelligence|\bAI\b|hyperscaler|backlog)", re.I)


def daily(ticker: str) -> pd.Series:
    rows = json.loads((DAILY / f"massive_{ticker}_daily.json").read_text())["results"]
    return pd.Series({pd.to_datetime(r["t"], unit="ms").normalize(): float(r["c"]) for r in rows}).sort_index()


def window_return(series: pd.Series, start: str, end: str) -> float:
    window = series.loc[start:end]
    return float(window.iloc[-1] / window.iloc[0] - 1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    news_rows = []
    for ticker in TICKERS:
        articles = json.loads((NEWS / f"massive_news_{ticker}.json").read_text())["results"]
        texts = [" ".join([a.get("title", ""), a.get("description") or "", " ".join(a.get("keywords") or [])]) for a in articles]
        news_rows.append({
            "ticker": ticker, "articles": len(articles),
            "enso_mentions": sum(bool(ENSO.search(text)) for text in texts),
            "cooling_ai_mentions": sum(bool(THEME.search(text)) for text in texts),
        })
    news = pd.DataFrame(news_rows)
    news["cooling_ai_share"] = news["cooling_ai_mentions"] / news["articles"]
    news.to_csv(OUT / "news_attention_pricing.csv", index=False)

    prices = {ticker: daily(ticker) for ticker in TICKERS + ["XLI"]}
    periods = [
        ("Apr 1 to Sep 11", "2026-04-01", "2026-09-11"),
        ("Jul 1 to Sep 11", "2026-07-01", "2026-09-11"),
        ("IRI Aug 19 two-day", "2026-08-18", "2026-08-20"),
        ("NOAA Sep 10 two-day", "2026-09-09", "2026-09-11"),
    ]
    rows = []
    for ticker in TICKERS:
        for name, start, end in periods:
            ret = window_return(prices[ticker], start, end)
            benchmark = window_return(prices["XLI"], start, end)
            rows.append({"ticker": ticker, "window": name, "start": start, "end": end,
                         "return": ret, "xli_return": benchmark, "abnormal_return": ret - benchmark})
    pd.DataFrame(rows).to_csv(OUT / "enso_news_event_windows.csv", index=False)
    print(news.to_string(index=False))


if __name__ == "__main__":
    main()
