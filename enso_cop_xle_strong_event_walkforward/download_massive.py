#!/usr/bin/env python3
"""Download Massive daily bars and dividends without logging credentials."""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CONFIG = ROOT.parent / "config.json"
TICKERS = ("COP", "XLE")
START = "2012-06-01"
END = "2026-09-11"


def request_json(url: str, key: str) -> dict:
    separator = "&" if "?" in url else "?"
    if "apiKey=" not in url:
        url = f"{url}{separator}{urllib.parse.urlencode({'apiKey': key})}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-backtest/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=60, context=context) as response:
        return json.load(response)


def download_paginated(url: str, key: str) -> dict:
    rows: list[dict] = []
    request_ids: list[str] = []
    first: dict | None = None
    while url:
        payload = request_json(url, key)
        first = first or payload
        rows.extend(payload.get("results", []))
        if payload.get("request_id"):
            request_ids.append(payload["request_id"])
        url = payload.get("next_url")
    return {
        "status": (first or {}).get("status", "UNKNOWN"),
        "ticker": (first or {}).get("ticker"),
        "adjusted": (first or {}).get("adjusted"),
        "resultsCount": len(rows),
        "request_ids": request_ids,
        "results": rows,
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    if not key:
        raise RuntimeError("Missing Massive API key in config.json")
    for ticker in TICKERS:
        bars_url = (
            f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{START}/{END}"
            "?adjusted=true&sort=asc&limit=50000"
        )
        bars = download_paginated(bars_url, key)
        if not bars["results"]:
            raise RuntimeError(f"No daily bars returned for {ticker}")
        (RAW / f"massive_{ticker}_daily.json").write_text(json.dumps(bars, separators=(",", ":")))

        div_url = (
            "https://api.massive.com/stocks/v1/dividends?"
            + urllib.parse.urlencode({
                "ticker": ticker,
                "ex_dividend_date.gte": START,
                "ex_dividend_date.lte": END,
                "sort": "ex_dividend_date.asc",
                "limit": 5000,
            })
        )
        divs = download_paginated(div_url, key)
        (RAW / f"massive_{ticker}_dividends.json").write_text(json.dumps(divs, separators=(",", ":")))
        print(f"{ticker}: {len(bars['results'])} daily bars, {len(divs['results'])} dividends")


if __name__ == "__main__":
    main()
