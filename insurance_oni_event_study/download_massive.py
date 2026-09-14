#!/usr/bin/env python3
"""Download split-adjusted daily aggregates for the supplied insurer universe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
INPUT = Path(os.environ.get(
    "INSURANCE_UNIVERSE_CSV",
    str(WORKSPACE / "inputs" / "insurance" / "el_nino_insurance_stocks_massive.csv"),
))
CONFIG = WORKSPACE / "config.json"
RAW = ROOT / "raw"
PREPARED = ROOT / "prepared"
BASE = "https://api.massive.com"
START = "2001-01-01"
END = "2026-09-12"


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.netloc and parts.netloc != "api.massive.com":
        raise RuntimeError("Unexpected pagination host")
    return urlunsplit(("https", "api.massive.com", parts.path, parts.query, ""))


class MassiveClient:
    def __init__(self, key: str) -> None:
        self.key = key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "insurance-oni-event-study/1.0"})

    def get(self, url: str, params: dict | None = None) -> dict:
        clean = safe_url(urljoin(BASE, url))
        query = dict(params or {})
        query["apiKey"] = self.key
        for attempt in range(7):
            try:
                response = self.session.get(clean, params=query, timeout=90)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 6:
                        response.raise_for_status()
                    time.sleep(min(60, 2**attempt) + random.random())
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 6:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    raise RuntimeError(f"Massive request failed (HTTP {status or 'network'})") from None
                time.sleep(min(30, 2**attempt) + random.random())
        raise AssertionError("unreachable")

    def pages(self, path: str, params: dict) -> tuple[list[dict], list[str]]:
        rows: list[dict] = []
        request_ids: list[str] = []
        url: str | None = path
        next_params: dict | None = params
        seen: set[str] = set()
        while url:
            clean = safe_url(urljoin(BASE, url))
            marker = clean.replace(self.key, "<redacted>")
            if marker in seen:
                raise RuntimeError("Repeated Massive pagination cursor")
            seen.add(marker)
            payload = self.get(clean, next_params)
            page = payload.get("results", [])
            if not isinstance(page, list):
                raise RuntimeError("Unexpected Massive response schema")
            rows.extend(page)
            if payload.get("request_id"):
                request_ids.append(payload["request_id"])
            url = payload.get("next_url")
            next_params = None
        return rows, request_ids


def save_json_gz(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(obj, handle, separators=(",", ":"))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PREPARED.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT)
    universe = (
        source[["ticker", "company_name", "insurance_role", "property_insurance",
                "natural_catastrophe_or_reinsurance", "agriculture_insurance"]]
        .drop_duplicates("ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    universe.to_csv(PREPARED / "company_universe.csv", index=False)
    # ACE became Chubb/CB in January 2016; RE became Everest/EG in July 2023.
    # KIE is retained as a sector benchmark and SPY as the broad-market benchmark.
    tickers = universe["ticker"].tolist() + ["ACE", "RE", "KIE", "SPY"]
    cfg = json.loads(CONFIG.read_text())
    client = MassiveClient(cfg["api_key"])
    manifest: list[dict] = []

    for number, ticker in enumerate(tickers, start=1):
        raw_path = RAW / f"massive_{ticker}_daily_{args.start}_{args.end}.json.gz"
        try:
            if raw_path.exists() and not args.refresh:
                with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                rows = payload["results"]
                source_status = "cached"
            else:
                path = f"/v2/aggs/ticker/{ticker}/range/1/day/{args.start}/{args.end}"
                rows, request_ids = client.pages(
                    path,
                    {"adjusted": "true", "sort": "asc", "limit": 50000},
                )
                payload = {
                    "ticker": ticker,
                    "adjusted": True,
                    "start": args.start,
                    "end": args.end,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "request_ids": request_ids,
                    "results": rows,
                }
                save_json_gz(raw_path, payload)
                source_status = "downloaded"

            if rows:
                dates = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True)
                first_date = str(dates.min().date())
                last_date = str(dates.max().date())
                status = "ok"
            else:
                first_date = last_date = None
                status = "empty"
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            manifest.append({
                "ticker": ticker,
                "status": status,
                "source_status": source_status,
                "rows": len(rows),
                "first_date": first_date,
                "last_date": last_date,
                "adjusted_for_splits": True,
                "includes_dividends": False,
                "raw_file": str(raw_path.relative_to(ROOT)),
                "sha256": digest,
                "error": "",
            })
            print(f"[{number:02d}/{len(tickers)}] {ticker}: {len(rows):,} rows, {first_date} to {last_date}", flush=True)
        except Exception as exc:
            manifest.append({
                "ticker": ticker,
                "status": "error",
                "source_status": "failed",
                "rows": 0,
                "first_date": None,
                "last_date": None,
                "adjusted_for_splits": True,
                "includes_dividends": False,
                "raw_file": "",
                "sha256": "",
                "error": str(exc),
            })
            print(f"[{number:02d}/{len(tickers)}] {ticker}: ERROR {exc}", flush=True)

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(PREPARED / "download_manifest.csv", index=False)
    summary = {
        "requested_start": args.start,
        "requested_end": args.end,
        "tickers_requested": len(tickers),
        "tickers_ok": int((manifest_frame["status"] == "ok").sum()),
        "tickers_empty": int((manifest_frame["status"] == "empty").sum()),
        "tickers_error": int((manifest_frame["status"] == "error").sum()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (PREPARED / "download_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
