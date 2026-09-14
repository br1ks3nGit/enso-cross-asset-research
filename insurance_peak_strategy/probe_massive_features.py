#!/usr/bin/env python3
"""Probe Massive datasets needed by the peak-El-Nino strategy."""

from __future__ import annotations

import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent / "config.json"
BASE = "https://api.massive.com"


def main() -> None:
    key = json.loads(CONFIG.read_text())["api_key"]
    session = requests.Session()
    session.headers["User-Agent"] = "insurance-peak-strategy/1.0"
    probes = {
        "short_interest": ("/stocks/v1/short-interest", {"ticker": "ROOT", "limit": 10}),
        "short_volume": ("/stocks/v1/short-volume", {"ticker": "ROOT", "limit": 10}),
        "ticker_float": ("/v3/reference/tickers/ROOT", {"date": "2023-09-01"}),
        "quotes": ("/v3/quotes/ROOT", {"timestamp.gte": "2023-09-01", "timestamp.lt": "2023-09-02", "limit": 10, "sort": "timestamp"}),
    }
    output = {}
    for name, (path, params) in probes.items():
        try:
            response = session.get(BASE + path, params={**params, "apiKey": key}, timeout=45)
            status = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"message": response.text[:300]}
            results = payload.get("results", []) if isinstance(payload, dict) else []
            sample = results[:2] if isinstance(results, list) else results
            output[name] = {
                "path": path,
                "params_without_key": params,
                "http_status": status,
                "payload_status": payload.get("status") if isinstance(payload, dict) else None,
                "message": payload.get("message") if isinstance(payload, dict) else None,
                "result_count": len(results) if isinstance(results, list) else (1 if results else 0),
                "sample": sample,
            }
            print(name, status, output[name]["result_count"], output[name]["message"])
        except requests.RequestException as exc:
            output[name] = {"path": path, "http_status": None, "error": type(exc).__name__}
            print(name, "network error")
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "raw" / "feature_probe.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
