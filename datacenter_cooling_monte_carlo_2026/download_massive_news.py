#!/usr/bin/env python3
"""Download recent Massive news for the cooling-equity universe."""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent / "config.json"
RAW = ROOT / "raw/news"
TICKERS = ("MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    context = ssl.create_default_context(cafile=certifi.where())
    for ticker in TICKERS:
        params = urllib.parse.urlencode({
            "ticker": ticker,
            "published_utc.gte": "2026-04-01",
            "published_utc.lte": "2026-09-12T23:59:59Z",
            "order": "desc",
            "sort": "published_utc",
            "limit": 1000,
            "apiKey": key,
        })
        req = urllib.request.Request(
            f"https://api.massive.com/v2/reference/news?{params}",
            headers={"User-Agent": "enso-cooling-news-study/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60, context=context) as response:
            payload = json.load(response)
        stored = {"status": payload.get("status"), "count": len(payload.get("results", [])),
                  "results": payload.get("results", [])}
        (RAW / f"massive_news_{ticker}.json").write_text(json.dumps(stored, separators=(",", ":")))
        print(f"{ticker}: {stored['count']} articles")


if __name__ == "__main__":
    main()
