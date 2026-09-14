#!/usr/bin/env python3
"""Select point-in-time COP/XLE option contracts and download Massive daily bars."""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PARENT = ROOT.parent
SOURCE = PARENT / "enso_cop_xle_strong_event_walkforward"
sys.path.insert(0, str(SOURCE))
from download_massive import download_paginated


def load_underlying_unadjusted(ticker: str, key: str) -> pd.DataFrame:
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-12-31"
        "?adjusted=false&sort=asc&limit=50000"
    )
    payload = download_paginated(url, key)
    (RAW / f"massive_{ticker}_daily_unadjusted_2024.json").write_text(json.dumps(payload, separators=(",", ":")))
    rows = payload["results"]
    return pd.DataFrame({
        "date": [pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize() for r in rows],
        "open": [float(r["o"]) for r in rows],
    }).set_index("date")


def fetch_chain(ticker: str, option_type: str, entry: pd.Timestamp, exit_date: pd.Timestamp, key: str) -> list[dict]:
    query = urllib.parse.urlencode({
        "underlying_ticker": ticker,
        "contract_type": option_type,
        "as_of": str(entry.date()),
        "expiration_date.gte": str((exit_date + pd.Timedelta(days=7)).date()),
        "expiration_date.lte": str((entry + pd.Timedelta(days=110)).date()),
        "limit": 1000,
        "sort": "expiration_date",
        "order": "asc",
    })
    payload = download_paginated("https://api.massive.com/v3/reference/options/contracts?" + query, key)
    filename = f"chain_{entry.date()}_{ticker}_{option_type}.json"
    (RAW / filename).write_text(json.dumps(payload, separators=(",", ":")))
    return payload["results"]


def select_liquid_pair(chain: list[dict], spot: float, option_type: str, entry: pd.Timestamp, exit_date: pd.Timestamp, key: str) -> tuple[dict, dict]:
    if not chain:
        raise RuntimeError("No option contracts returned")
    frame = pd.DataFrame(chain)
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"])
    frame["dte"] = (frame["expiration_date"] - entry).dt.days
    expiry = frame.loc[(frame["dte"] - 75).abs().idxmin(), "expiration_date"]
    same = frame[frame["expiration_date"] == expiry].copy()
    same["strike_price"] = same["strike_price"].astype(float)
    def pick(pool: pd.DataFrame, target: float) -> dict:
        ordered = pool.assign(distance=(pool["strike_price"] - target).abs()).sort_values(["distance", "strike_price"])
        for _, candidate in ordered.head(8).iterrows():
            contract = candidate.to_dict()
            payload = fetch_option_bars(contract["ticker"], entry, exit_date, key)
            dates = {pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize() for r in payload["results"]}
            if entry in dates and exit_date in dates:
                return contract
        raise RuntimeError(f"No liquid contract with entry and exit bars near strike {target:.2f}")

    atm = pick(same, spot)
    target = spot * (1.10 if option_type == "call" else 0.90)
    if option_type == "call":
        wing_pool = same[same["strike_price"] > float(atm["strike_price"])]
    else:
        wing_pool = same[same["strike_price"] < float(atm["strike_price"])]
    if wing_pool.empty:
        raise RuntimeError("No valid vertical-spread wing")
    wing = pick(wing_pool, target)
    return atm, wing


def fetch_option_bars(contract: str, entry: pd.Timestamp, exit_date: pd.Timestamp, key: str) -> dict:
    url = (
        f"https://api.massive.com/v2/aggs/ticker/{contract}/range/1/day/{entry.date()}/{exit_date.date()}"
        "?adjusted=true&sort=asc&limit=50000"
    )
    payload = download_paginated(url, key)
    safe = contract.replace(":", "_")
    (RAW / f"bars_{safe}.json").write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    key = json.loads((PARENT / "config.json").read_text())["api_key"]
    trades = pd.read_csv(SOURCE / "calculated" / "non_overlapping_trades.csv", parse_dates=["entry_date", "exit_date"])
    trades = trades[(trades["rule"] == "S5 Strong but cooling") & (trades["horizon_days"] == 40)].copy()
    underlyings = {ticker: load_underlying_unadjusted(ticker, key) for ticker in ("XLE", "COP")}
    selections = []
    for event_id, (_, trade) in enumerate(trades.iterrows(), start=1):
        entry, exit_date = trade["entry_date"], trade["exit_date"]
        for ticker, option_type in (("XLE", "call"), ("COP", "put")):
            spot = float(underlyings[ticker].loc[entry, "open"])
            chain = fetch_chain(ticker, option_type, entry, exit_date, key)
            atm, wing = select_liquid_pair(chain, spot, option_type, entry, exit_date, key)
            for role, contract in (("atm", atm), ("wing", wing)):
                row = {
                    "event_id": event_id, "entry_date": entry, "planned_exit_date": exit_date,
                    "underlying": ticker, "option_type": option_type, "role": role,
                    "spot_open": spot, "contract": contract["ticker"],
                    "expiration_date": contract["expiration_date"],
                    "strike": float(contract["strike_price"]),
                    "shares_per_contract": int(contract.get("shares_per_contract", 100)),
                }
                selections.append(row)
                payload = json.loads((RAW / f"bars_{contract['ticker'].replace(':', '_')}.json").read_text())
                print(f"{entry.date()} {contract['ticker']}: {len(payload['results'])} daily bars")
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(ROOT / "contract_selections.csv", index=False)
    print(selection_frame.to_string(index=False))


if __name__ == "__main__":
    main()
