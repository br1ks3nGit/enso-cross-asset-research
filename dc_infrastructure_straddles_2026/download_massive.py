#!/usr/bin/env python3
"""Download point-in-time Massive data for the 2026 data-center straddle test."""

from __future__ import annotations

import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CONFIG = ROOT.parent / "config.json"
TICKERS = ("VRT", "MOD", "ETN", "PWR", "GEV")
START = "2026-04-01"
END = "2026-09-11"
TARGET_DTE = 55
MIN_DTE = 45
MAX_DTE = 70
TARGET_HOLDING_SESSIONS = 20


def request_json(url: str, key: str) -> dict:
    separator = "&" if "?" in url else "?"
    if "apiKey=" not in url:
        url = f"{url}{separator}{urllib.parse.urlencode({'apiKey': key})}"
    req = urllib.request.Request(url, headers={"User-Agent": "dc-infrastructure-straddle-backtest/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=90, context=context) as response:
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


def save_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_daily(ticker: str, key: str) -> pd.DataFrame:
    safe = ticker.replace(":", "_")
    path = RAW / f"massive_{safe}_daily.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        url = (
            f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{START}/{END}"
            "?adjusted=false&sort=asc&limit=50000"
        )
        payload = download_paginated(url, key)
        save_payload(path, payload)
    if not payload.get("results"):
        raise RuntimeError(f"No daily bars for {ticker}")
    rows = payload["results"]
    frame = pd.DataFrame({
        "date": [pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize() for r in rows],
        "open": [float(r["o"]) for r in rows],
        "close": [float(r["c"]) for r in rows],
        "volume": [float(r.get("v", 0)) for r in rows],
    })
    return frame.set_index("date").sort_index()


def fetch_chain(ticker: str, entry: pd.Timestamp, spot: float, key: str) -> list[dict]:
    filename = RAW / f"chain_{entry.date()}_{ticker}.json"
    if filename.exists():
        return json.loads(filename.read_text())["results"]
    query = urllib.parse.urlencode({
        "underlying_ticker": ticker,
        "as_of": str(entry.date()),
        "expiration_date.gte": str((entry + pd.Timedelta(days=MIN_DTE)).date()),
        "expiration_date.lte": str((entry + pd.Timedelta(days=MAX_DTE)).date()),
        "strike_price.gte": f"{spot * 0.70:.4f}",
        "strike_price.lte": f"{spot * 1.30:.4f}",
        "limit": 1000,
        "sort": "expiration_date",
        "order": "asc",
    })
    payload = download_paginated("https://api.massive.com/v3/reference/options/contracts?" + query, key)
    save_payload(filename, payload)
    return payload["results"]


def fetch_option_bars(contract: str, entry: pd.Timestamp, exit_date: pd.Timestamp, key: str) -> dict:
    safe = contract.replace(":", "_")
    path = RAW / f"bars_{safe}_{entry.date()}_{exit_date.date()}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{contract}/range/1/day/{entry.date()}/{exit_date.date()}"
        "?adjusted=true&sort=asc&limit=50000"
    )
    payload = download_paginated(url, key)
    save_payload(path, payload)
    return payload


def bar_on(payload: dict, date: pd.Timestamp) -> dict | None:
    for row in payload.get("results", []):
        row_date = pd.to_datetime(row["t"], unit="ms", utc=True).tz_convert(None).normalize()
        if row_date == date:
            return row
    return None


def select_straddle(
    ticker: str,
    entry: pd.Timestamp,
    exit_date: pd.Timestamp,
    spot: float,
    key: str,
) -> dict | None:
    chain = fetch_chain(ticker, entry, spot, key)
    if not chain:
        return None
    frame = pd.DataFrame(chain)
    needed = {"ticker", "contract_type", "expiration_date", "strike_price"}
    if not needed.issubset(frame.columns):
        return None
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"])
    frame["strike_price"] = frame["strike_price"].astype(float)
    frame["dte"] = (frame["expiration_date"] - entry).dt.days
    calls = frame[frame["contract_type"] == "call"]
    puts = frame[frame["contract_type"] == "put"]
    expiries = sorted(
        set(calls["expiration_date"]).intersection(puts["expiration_date"]),
        key=lambda x: (abs((x - entry).days - TARGET_DTE), x),
    )
    for expiry in expiries:
        call_exp = calls[calls["expiration_date"] == expiry]
        put_exp = puts[puts["expiration_date"] == expiry]
        strikes = sorted(
            set(call_exp["strike_price"]).intersection(put_exp["strike_price"]),
            key=lambda x: (abs(x - spot), x),
        )
        for strike in strikes[:16]:
            call = call_exp[call_exp["strike_price"] == strike].iloc[0]
            put = put_exp[put_exp["strike_price"] == strike].iloc[0]
            call_payload = fetch_option_bars(call["ticker"], entry, exit_date, key)
            put_payload = fetch_option_bars(put["ticker"], entry, exit_date, key)
            call_entry, call_exit = bar_on(call_payload, entry), bar_on(call_payload, exit_date)
            put_entry, put_exit = bar_on(put_payload, entry), bar_on(put_payload, exit_date)
            if all(x is not None for x in (call_entry, call_exit, put_entry, put_exit)):
                return {
                    "ticker": ticker,
                    "entry_date": entry,
                    "exit_date": exit_date,
                    "target_holding_sessions": TARGET_HOLDING_SESSIONS,
                    "spot_open": spot,
                    "expiration_date": expiry,
                    "dte_at_entry": (expiry - entry).days,
                    "strike": strike,
                    "call_contract": call["ticker"],
                    "put_contract": put["ticker"],
                    "call_entry_open": float(call_entry["o"]),
                    "put_entry_open": float(put_entry["o"]),
                    "call_exit_close": float(call_exit["c"]),
                    "put_exit_close": float(put_exit["c"]),
                    "call_entry_volume": int(call_entry.get("v", 0)),
                    "put_entry_volume": int(put_entry.get("v", 0)),
                    "call_exit_volume": int(call_exit.get("v", 0)),
                    "put_exit_volume": int(put_exit.get("v", 0)),
                }
    return None


def decision_schedule(common_sessions: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int, bool]]:
    out = []
    for month in pd.period_range("2026-04", "2026-09", freq="M"):
        sessions = common_sessions[common_sessions.to_period("M") == month]
        if len(sessions) == 0:
            continue
        entry = sessions[0]
        pos = common_sessions.get_loc(entry)
        target_pos = pos + TARGET_HOLDING_SESSIONS - 1
        partial = target_pos >= len(common_sessions)
        exit_date = common_sessions[min(target_pos, len(common_sessions) - 1)]
        actual_sessions = common_sessions.get_loc(exit_date) - pos + 1
        out.append((entry, exit_date, actual_sessions, partial))
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads(CONFIG.read_text())["api_key"]
    if not key:
        raise RuntimeError("Missing Massive API key in config.json")
    underlying = {ticker: load_daily(ticker, key) for ticker in TICKERS}
    fx = load_daily("C:USDHKD", key)
    common = underlying[TICKERS[0]].index
    for ticker in TICKERS[1:]:
        common = common.intersection(underlying[ticker].index)
    schedule = decision_schedule(common)
    selections: list[dict] = []
    failures: list[dict] = []
    for ticker in TICKERS:
        for entry, exit_date, actual_sessions, partial in schedule:
            spot = float(underlying[ticker].loc[entry, "open"])
            selected = select_straddle(ticker, entry, exit_date, spot, key)
            if selected is None:
                failures.append({
                    "ticker": ticker, "entry_date": entry, "exit_date": exit_date,
                    "reason": "No same-strike call/put pair with exact entry and exit trade bars",
                })
                print(f"MISS {ticker} {entry.date()} -> {exit_date.date()}")
                continue
            selected["actual_holding_sessions"] = actual_sessions
            selected["partial_window"] = partial
            selections.append(selected)
            print(
                f"OK {ticker} {entry.date()} {selected['expiration_date'].date()} "
                f"K={selected['strike']:.2f} -> {exit_date.date()}"
            )
    frame = pd.DataFrame(selections)
    frame.to_csv(ROOT / "contract_selections.csv", index=False)
    pd.DataFrame(failures).to_csv(ROOT / "selection_failures.csv", index=False)
    fx.reset_index().to_csv(ROOT / "usd_hkd_daily.csv", index=False)
    print(f"Selected {len(frame)} of {len(TICKERS) * len(schedule)} ticker-months")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
