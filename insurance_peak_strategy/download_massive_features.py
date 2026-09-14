#!/usr/bin/env python3
"""Download Massive short-sale, reference, and weekly NBBO data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import random
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent / "config.json"
RAW = ROOT / "raw"
PRICE_RAW = ROOT.parent / "insurance_oni_event_study" / "raw"
TICKERS = ("ROOT", "PLMR", "KNSL", "MCY", "LMND", "CINF", "PGR", "AIZ", "AIG", "BAP", "ALL", "HIG")
WINDOWS = (
    (date(2010, 3, 1), date(2010, 4, 30)),
    (date(2015, 11, 1), date(2016, 6, 30)),
    (date(2023, 12, 1), date(2024, 5, 31)),
)
BASE = "https://api.massive.com"


def clean_url(value: str) -> str:
    parts = urlsplit(urljoin(BASE, value))
    if parts.netloc != "api.massive.com":
        raise RuntimeError("Unexpected Massive pagination host")
    query = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in {"apikey", "api_key"}]
    return urlunsplit(("https", parts.netloc, parts.path, urlencode(query), ""))


class Client:
    def __init__(self, key: str) -> None:
        self.key = key

    def get(self, url: str, params: dict | None = None) -> dict:
        clean = clean_url(url)
        query = dict(params or {})
        query["apiKey"] = self.key
        for attempt in range(7):
            try:
                response = requests.get(clean, params=query, headers={"User-Agent": "insurance-peak-strategy/1.0"}, timeout=60)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 6:
                        response.raise_for_status()
                    time.sleep(min(40, 2**attempt) + random.random())
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"Massive request failed (HTTP {response.status_code})")
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 6:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    raise RuntimeError(f"Massive request failed (HTTP {status or 'network'})") from None
                time.sleep(min(30, 2**attempt) + random.random())
        raise AssertionError("unreachable")

    def pages(self, path: str, params: dict) -> list[dict]:
        rows: list[dict] = []
        url: str | None = path
        current: dict | None = params
        seen: set[str] = set()
        while url:
            marker = clean_url(url)
            if marker in seen:
                raise RuntimeError("Repeated Massive pagination cursor")
            seen.add(marker)
            payload = self.get(marker, current)
            result = payload.get("results", [])
            if isinstance(result, list):
                rows.extend(result)
            elif result:
                rows.append(result)
            url = payload.get("next_url")
            current = None
        return rows


def save_gz(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(obj, handle, separators=(",", ":"))
    tmp.replace(path)


def week_ranges() -> list[tuple[date, date]]:
    weeks: set[tuple[date, date]] = set()
    for start, end in WINDOWS:
        cursor = start - timedelta(days=start.weekday() + 7)
        while cursor <= end:
            weeks.add((cursor, cursor + timedelta(days=7)))
            cursor += timedelta(days=7)
    return sorted(weeks)


def load_price_dates(ticker: str) -> list[date]:
    matches = sorted(PRICE_RAW.glob(f"massive_{ticker}_daily_*.json.gz"))
    if not matches:
        return []
    with gzip.open(matches[-1], "rt", encoding="utf-8") as handle:
        rows = json.load(handle).get("results", [])
    return [datetime.fromtimestamp(row["t"] / 1000, timezone.utc).date() for row in rows]


def quote_job(client: Client, ticker: str, session_date: date, refresh: bool) -> dict:
    path = RAW / "quotes_weekly_nbbo" / ticker / f"{session_date}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {"ticker": ticker, "date": str(session_date), "rows": len(payload.get("results", [])), "cached": True}
    ny = ZoneInfo("America/New_York")
    start_dt = datetime.combine(session_date, dtime(15, 55), ny).astimezone(timezone.utc)
    stop_dt = datetime.combine(session_date, dtime(16, 0), ny).astimezone(timezone.utc)
    start_ns = int(start_dt.timestamp() * 1_000_000_000)
    stop_ns = int(stop_dt.timestamp() * 1_000_000_000)
    payload = client.get(
        f"/v3/quotes/{ticker}",
        {"timestamp.gte": start_ns, "timestamp.lt": stop_ns, "sort": "timestamp", "order": "desc", "limit": 1},
    )
    result = payload.get("results", [])
    rows = result if isinstance(result, list) else ([result] if result else [])
    save_gz(path, {"ticker": ticker, "session_date": str(session_date), "window_new_york": "15:55:00-16:00:00 (exclusive)", "results": rows[:1]})
    return {"ticker": ticker, "date": str(session_date), "rows": len(rows[:1]), "cached": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    client = Client(key)
    manifest: list[dict] = []

    for ticker in TICKERS:
        for dataset, endpoint, start_field, end_field in (
            ("short_interest", "/stocks/v1/short-interest", "settlement_date.gte", "settlement_date.lte"),
            ("short_volume", "/stocks/v1/short-volume", "date.gte", "date.lte"),
        ):
            target = RAW / dataset / f"{ticker}.json.gz"
            try:
                if target.exists() and not args.refresh:
                    with gzip.open(target, "rt", encoding="utf-8") as handle:
                        rows = json.load(handle).get("results", [])
                    cached = True
                else:
                    rows = client.pages(endpoint, {"ticker": ticker, start_field: "2009-01-01", end_field: "2024-06-30", "limit": 50000})
                    save_gz(target, {"ticker": ticker, "dataset": dataset, "results": rows})
                    cached = False
                manifest.append({"dataset": dataset, "ticker": ticker, "rows": len(rows), "status": "ok", "cached": cached})
                print(f"{dataset} {ticker}: {len(rows):,}", flush=True)
            except Exception as exc:
                manifest.append({"dataset": dataset, "ticker": ticker, "rows": 0, "status": "error", "cached": False, "error": str(exc)})
                print(f"{dataset} {ticker}: ERROR {exc}", flush=True)

        for snapshot in ("2010-03-01", "2015-11-01", "2023-12-01"):
            target = RAW / "reference" / f"{ticker}_{snapshot}.json.gz"
            try:
                if target.exists() and not args.refresh:
                    with gzip.open(target, "rt", encoding="utf-8") as handle:
                        rows = json.load(handle).get("results", [])
                    cached = True
                else:
                    rows = client.pages(f"/v3/reference/tickers/{ticker}", {"date": snapshot})
                    save_gz(target, {"ticker": ticker, "date": snapshot, "results": rows})
                    cached = False
                manifest.append({"dataset": "reference", "ticker": ticker, "date": snapshot, "rows": len(rows), "status": "ok", "cached": cached})
            except Exception as exc:
                manifest.append({"dataset": "reference", "ticker": ticker, "date": snapshot, "rows": 0, "status": "error", "cached": False, "error": str(exc)})

    jobs = []
    for ticker in TICKERS:
        available = set(load_price_dates(ticker))
        for a, b in week_ranges():
            sessions = sorted(day for day in available if a <= day < b)
            if sessions:
                jobs.append((ticker, sessions[-1]))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(quote_job, client, ticker, session_date, args.refresh): (ticker, session_date) for ticker, session_date in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            ticker, start = futures[future]
            try:
                row = future.result()
                manifest.append({"dataset": "quotes_weekly_nbbo", **row, "status": "ok"})
            except Exception as exc:
                manifest.append({"dataset": "quotes_weekly_nbbo", "ticker": ticker, "date": str(start), "rows": 0, "status": "error", "error": str(exc)})
            if number % 50 == 0 or number == len(jobs):
                print(f"weekly quotes: {number}/{len(jobs)}", flush=True)

    import pandas as pd
    frame = pd.DataFrame(manifest)
    frame.to_csv(ROOT / "feature_download_manifest.csv", index=False)
    print(frame.groupby(["dataset", "status"])["rows"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
