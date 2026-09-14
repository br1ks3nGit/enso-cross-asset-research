#!/usr/bin/env python3
"""Download point-in-time Massive news for the strategy universe by calendar year."""

from __future__ import annotations

import concurrent.futures
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/news"
CONFIG = ROOT.parent / "config.json"
TICKERS = ("MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC")
YEARS = range(2020, 2027)


def fetch(ticker: str, year: int, key: str) -> tuple[str, int, list[dict]]:
    end = f"{year}-12-31T23:59:59Z" if year < 2026 else "2026-09-12T23:59:59Z"
    params = urllib.parse.urlencode({
        "ticker": ticker, "published_utc.gte": f"{year}-01-01",
        "published_utc.lte": end, "order": "asc", "sort": "published_utc",
        "limit": 1000, "apiKey": key,
    })
    request = urllib.request.Request(
        f"https://api.massive.com/v2/reference/news?{params}",
        headers={"User-Agent": "enso-cooling-confirmed-strategy/1.0"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        payload = json.load(response)
    return ticker, year, payload.get("results", [])


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    jobs = [(ticker, year, key) for ticker in TICKERS for year in YEARS]
    combined = {ticker: [] for ticker in TICKERS}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            ticker, year, rows = future.result()
            combined[ticker].extend(rows)
            print(f"{ticker} {year}: {len(rows)}")
    for ticker, rows in combined.items():
        unique = {}
        for row in rows:
            unique[row.get("id") or row.get("article_url")] = row
        ordered = sorted(unique.values(), key=lambda row: row.get("published_utc", ""))
        payload = {"ticker": ticker, "count": len(ordered), "results": ordered}
        (RAW / f"massive_news_{ticker}_2020_2026.json").write_text(json.dumps(payload, separators=(",", ":")))
        print(f"{ticker} combined: {len(ordered)}")


if __name__ == "__main__":
    main()
