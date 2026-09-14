#!/usr/bin/env python3
"""Download adjusted monthly bars from Massive without logging credentials."""

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
TICKERS = ("MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC", "XLI")
START = "2004-01-01"
END = "2026-09-11"


def request_json(url: str, key: str) -> dict:
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}{urllib.parse.urlencode({'apiKey': key})}"
    request = urllib.request.Request(url, headers={"User-Agent": "enso-cooling-monte-carlo/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return json.load(response)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    if not key:
        raise RuntimeError("Missing Massive API key in config.json")
    for ticker in TICKERS:
        url = (
            f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/month/{START}/{END}"
            "?adjusted=true&sort=asc&limit=50000"
        )
        payload = request_json(url, key)
        rows = payload.get("results", [])
        if not rows:
            raise RuntimeError(f"No monthly bars returned for {ticker}: {payload.get('status')}")
        stored = {
            "status": payload.get("status"),
            "ticker": payload.get("ticker"),
            "adjusted": payload.get("adjusted"),
            "resultsCount": len(rows),
            "request_id": payload.get("request_id"),
            "results": rows,
        }
        (RAW / f"massive_{ticker}_monthly.json").write_text(json.dumps(stored, separators=(",", ":")))
        print(f"{ticker}: {len(rows)} adjusted monthly bars")


if __name__ == "__main__":
    main()
