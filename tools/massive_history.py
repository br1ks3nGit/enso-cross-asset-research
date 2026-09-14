#!/usr/bin/env python3
"""Historical Massive ingestion. Python 3.10+; standard library only.

No credentials are included. Default mode stores individual trades, preserving
all vendor fields and integer nanoseconds. See README.md before a large backfill.
"""
import argparse
from collections import Counter
from datetime import date, datetime, time as dtime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
from zoneinfo import ZoneInfo

BASE = "https://api.massive.com"
UTC = timezone.utc
DAY = timedelta(days=1)
SCHEMA = 1


class APIError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def clean_url(value):
    """Never persist a vendor-supplied API key or forward it to another host."""
    p = urlsplit(urljoin(BASE, value))
    if p.scheme != "https" or p.netloc != "api.massive.com":
        raise APIError("Unexpected API/pagination host")
    query = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in {"apikey", "api_key"}]
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(query), ""))


class Client:
    def __init__(self, key, rpm=5, retries=6):
        if not key.strip():
            raise ValueError('Enter your key in config.json: "api_key": "" or MASSIVE_API_KEY.')
        self.key, self.interval, self.retries = key, 60 / rpm, retries
        self.last = 0.0
        self.opener = build_opener(NoRedirect())

    def get(self, path, params=None):
        url = clean_url(path)
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params)
        for attempt in range(self.retries + 1):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            req = Request(url, headers={"Authorization": "Bearer " + self.key,
                                       "User-Agent": "historical-research/1.0"})
            delay = min(60, 2 ** attempt) + random.random()
            try:
                with self.opener.open(req, timeout=90) as response:
                    body = json.load(response)
                if not isinstance(body, dict):
                    raise APIError("API returned an unexpected JSON structure")
                if body.get("status") in {"ERROR", "NOT_AUTHORIZED", "DELAYED"} or body.get("error"):
                    raise APIError("API returned a non-success status; check access and request parameters")
                return body
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After")
                exc.close()
                if status not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise APIError(f"HTTP {status}; check symbol, date coverage, endpoint and entitlement", status) from None
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds())
                        except (ValueError, TypeError):
                            pass
            except (URLError, TimeoutError, OSError, json.JSONDecodeError):
                if attempt == self.retries:
                    raise APIError("Network or JSON response failure after retries") from None
            time.sleep(delay)

    def pages(self, path, params=None):
        seen = set()
        while path:
            marker = clean_url(path) + json.dumps(params, sort_keys=True)
            if marker in seen:
                raise APIError("Repeated pagination cursor; refusing a partial download")
            seen.add(marker)
            body = self.get(path, params)
            rows = body.get("results", [])
            if not isinstance(rows, list):
                raise APIError("Expected an array of results")
            yield rows
            path, params = body.get("next_url"), None


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def chunks(start, stop, days):
    """Nonoverlapping half-open date intervals."""
    while start < stop:
        nxt = min(stop, start + timedelta(days=days))
        yield start, nxt
        start = nxt


def ns_at(day, tz="UTC"):
    dt = datetime.combine(day, dtime(), ZoneInfo(tz)).astimezone(UTC)
    delta = dt - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000


def iso_ns(ns):
    seconds, remainder = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{remainder:09d}Z"


def normalize(row, asset, mode, instrument):
    field = "t" if asset == "stocks" and mode == "bars" else (
        "sip_timestamp" if asset == "stocks" else "window_start" if mode == "bars" else "timestamp")
    stamp = row.get(field)
    if not isinstance(stamp, int):
        raise APIError(f"Missing integer {field}; schema changed")
    stamp *= 1_000_000 if field == "t" else 1
    out = {"schema_version": SCHEMA, "asset_class": asset, "dataset": mode,
           "instrument": instrument, "timestamp_ns": stamp, "timestamp_utc": iso_ns(stamp),
           "timestamp_basis": field, "raw": row}
    if mode == "bars":
        mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                   "vw": "vwap", "n": "transactions"} if asset == "stocks" else {
                       k: k for k in ("open", "high", "low", "close", "volume", "transactions", "settlement_price", "dollar_volume")}
        out.update({dest: row[src] for src, dest in mapping.items() if src in row})
        if asset == "futures" and row.get("volume") and "dollar_volume" in row:
            out["vwap"] = row["dollar_volume"] / row["volume"]
    else:
        out.update({k: row[k] for k in ("price", "size") if k in row})
    if "session_end_date" in row:
        out["session_end_date"] = row["session_end_date"]
    return out


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "manifest.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, spec TEXT, status TEXT, rows INTEGER, first_ns INTEGER, last_ns INTEGER, file TEXT, error TEXT, updated TEXT)")

    def save(self, spec, status, count=0, first=None, last=None, file=None, error=None):
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?)", (
            key, json.dumps(spec, sort_keys=True), status, count, first, last, file, error, datetime.now(UTC).isoformat()))
        self.db.commit()
        return key

    def download(self, client, spec, retry_empty=False):
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        existing = self.db.execute("SELECT status, file FROM jobs WHERE id=?", (key,)).fetchone()
        if existing and existing[0] in {"complete", "empty"} and (self.root / existing[1]).is_file():
            if existing[0] != "empty" or not retry_empty:
                return
        rel = Path(spec["asset"]) / spec["mode"] / quote(spec["symbol"], safe="") / (spec["start"] + "_" + key[:16] + ".jsonl.gz")
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".partial")
        count, first, last = 0, None, None
        self.save(spec, "running")
        try:
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as stream:
                for page in client.pages(spec["path"], spec["params"]):
                    # Keep distinct trades at the same timestamp: never deduplicate on time.
                    for raw in page:
                        row = normalize(raw, spec["asset"], spec["mode"], spec["instrument"])
                        if spec["mode"] == "bars":
                            row["bar_size"] = spec["params"].get("resolution") or spec["path"].split("/range/1/")[1].split("/")[0]
                            row["split_adjusted"] = spec["params"].get("adjusted") == "true" if spec["asset"] == "stocks" else None
                        stamp = row["timestamp_ns"]
                        if not spec["lower_ns"] <= stamp < spec["upper_ns"]:
                            continue
                        if spec.get("session_start") and not (spec["session_start"] <= row.get("session_end_date", "") <= spec["session_end"]):
                            continue
                        first = stamp if first is None else min(first, stamp)
                        last = stamp if last is None else max(last, stamp)
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                        count += 1
            tmp.replace(target)
            self.save(spec, "complete" if count else "empty", count, first, last, str(rel))
            print(f'{spec["symbol"]} {spec["start"]}: {count:,} records', flush=True)
        except (APIError, OSError) as exc:
            self.save(spec, "error", error=str(exc))
            raise

    def report(self):
        rows = self.db.execute("SELECT spec,status,rows,first_ns,last_ns,file,error FROM jobs ORDER BY id")
        counts = Counter()
        with (self.root / "coverage.jsonl").open("w") as out:
            for spec, status, count, first, last, file, error in rows:
                counts[status] += 1
                out.write(json.dumps({"request": json.loads(spec), "status": status, "rows": count,
                    "first_utc": iso_ns(first) if first is not None else None,
                    "last_utc": iso_ns(last) if last is not None else None, "file": file, "error": error}) + "\n")
        atomic_json(self.root / "coverage_summary.json", dict(counts))
        return dict(counts)


AG_WORDS = re.compile(r"\b(agricultur\w*|grain\w*|oilseed\w*|livestock|dairy|softs|corn|wheat|soy\w*|oats?|rice|canola|rapeseed|palm|cattle|hogs?|pork|milk|cheese|butter|whey|coffee|cocoa|cotton|sugar|lumber|orange juice|fertilizer|urea)\b", re.I)
METALS = re.compile(r"\b(alumin(?:i)?um|copper|nickel)\b", re.I)


def product_reason(p, config):
    code = p.get("product_code")
    if code in config["exclude_product_codes"] or p.get("type") == "combo":
        return None
    if code in config["include_product_codes"]:
        return "explicit_include"
    description = " ".join(str(p.get(k, "")) for k in ("name", "asset_class", "asset_sub_class", "sector", "sub_sector")).replace("_", " ")
    if AG_WORDS.search(description):
        return "agriculture_candidate"
    if METALS.search(description):
        return "requested_metal_candidate"
    return None


def cached_rows(client, root, path, params):
    digest = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    dest = root / "reference" / (digest + ".json")
    if dest.exists():
        return json.loads(dest.read_text())["results"]
    rows = [row for page in client.pages(path, params) for row in page]
    atomic_json(dest, {"path": path, "params": params, "retrieved_at": datetime.now(UTC).isoformat(), "results": rows})
    return rows


def discover(client, config, root):
    start = max(date.fromisoformat(config["start"]), date.fromisoformat(config["futures_history_start"]))
    end = date.fromisoformat(config["end"])
    products, contracts = {}, {}
    # Historical snapshots catch discontinued products; retain all returned metadata.
    snapshots = {a for a, _ in chunks(start, end + DAY, config["product_snapshot_days"])}
    if start <= end:
        snapshots.add(end)
    for at in sorted(snapshots):
        for p in cached_rows(client, root, "/futures/v1/products", {"date": str(at), "limit": 1000}):
            code = p.get("product_code")
            if code:
                record = products.setdefault(code, {"product": p, "selected": False, "reason": None})
                reason = product_reason(p, config)
                if reason:
                    record.update(product=p, selected=True, reason=reason)
        print(f"Product catalogue: {at}", flush=True)
    atomic_json(root / "products.json", list(products.values()))
    for code, entry in sorted(products.items()):
        if not entry["selected"]:
            continue
        params = {"product_code": code, "first_trade_date.lte": str(end),
                  "last_trade_date.gte": str(start), "limit": 1000, "sort": "last_trade_date.asc"}
        # No active=true filter: expired contracts are necessary for historical research.
        for c in cached_rows(client, root, "/futures/v1/contracts", params):
            if c.get("type") == "combo" or not c.get("ticker"):
                continue
            # One-digit expiry tickers can repeat across decades.
            key = (c["ticker"], c.get("first_trade_date"), c.get("last_trade_date"))
            contracts[key] = {"contract": c, "product": entry["product"]}
    result = {"start": config["start"], "end": config["end"], "config_fingerprint": discovery_fingerprint(config),
              "contracts": list(contracts.values()),
              "limitations": ["Product classification is a candidate screen; inspect products.json.",
                  "31-day snapshots can miss very short-lived products; set product_snapshot_days=1 for a daily scan.",
                  "CME-group catalogue is not all global agriculture or LME metals.",
                  "Contract reference completeness and pre-2017 history require vendor confirmation."]}
    atomic_json(root / "universe.json", result)
    return result


def discovery_fingerprint(config):
    fields = ("start", "end", "futures_history_start", "product_snapshot_days", "include_product_codes", "exclude_product_codes")
    return hashlib.sha256(json.dumps({k: config[k] for k in fields}, sort_keys=True).encode()).hexdigest()


def request_spec(config, asset, ticker, instrument, a, b):
    mode = config["mode"]
    tz = "America/New_York" if asset == "stocks" else "UTC"
    lo, hi = ns_at(a, tz), ns_at(b, tz)
    ticker_path = quote(ticker, safe="")
    if mode == "trades":
        path = f"/v3/trades/{ticker_path}" if asset == "stocks" else f"/futures/v1/trades/{ticker_path}"
        params = {"timestamp.gte": lo, "timestamp.lt": hi, "limit": 50000}
        params.update({"sort": "timestamp", "order": "asc"} if asset == "stocks" else {"sort": "timestamp.asc"})
    elif asset == "stocks":
        path = f'/v2/aggs/ticker/{ticker_path}/range/1/{config["stock_bar_size"]}/{lo // 1_000_000}/{hi // 1_000_000 - 1}'
        params = {"adjusted": str(config["stock_adjusted"]).lower(), "sort": "asc", "limit": 50000}
    else:
        path = f"/futures/v1/aggs/{ticker_path}"
        params = {"resolution": config["futures_bar_size"], "window_start.gte": lo,
                  "window_start.lt": hi, "sort": "window_start.asc", "limit": 50000}
    return {"schema_version": SCHEMA, "asset": asset, "mode": mode, "symbol": ticker,
            "instrument": instrument, "start": str(a), "end_exclusive": str(b), "lower_ns": lo,
            "upper_ns": hi, "path": path, "params": params}


def jobs(config, universe, asset):
    start, stop = date.fromisoformat(config["start"]), date.fromisoformat(config["end"]) + DAY
    if asset in {"all", "stocks"}:
        for equity in config["equities"]:
            if not equity.get("enabled", True):
                continue
            for seg in equity["segments"]:
                a = max(start, date.fromisoformat(seg.get("start", str(start))))
                b = min(stop, date.fromisoformat(seg.get("end", str(stop - DAY))) + DAY)
                for left, right in chunks(a, b, 1 if config["mode"] == "trades" else 7):
                    yield request_spec(config, "stocks", seg["ticker"], {"symbol": equity["symbol"], **seg}, left, right)
    if asset in {"all", "futures"}:
        floor = date.fromisoformat(config["futures_history_start"])
        for item in universe["contracts"]:
            c = item["contract"]
            first = date.fromisoformat(c.get("first_trade_date") or str(start))
            last = date.fromisoformat(c.get("last_trade_date") or str(stop - DAY))
            # Include the overnight opening of the first trade session.
            a, b = max(start, floor, first - DAY), min(stop, last + DAY)
            session_mode = config["mode"] == "bars" and config["futures_bar_size"] == "1session"
            if session_mode:
                a = max(start, floor, first) - DAY
            for left, right in chunks(a, b, 1 if config["mode"] == "trades" else 7):
                spec = request_spec(config, "futures", c["ticker"], item, left, right)
                if session_mode:
                    spec.update(session_start=str(max(start, first)), session_end=str(min(stop - DAY, last)))
                yield spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "discover", "download", "report"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--asset", choices=["all", "stocks", "futures"], default="all")
    parser.add_argument("--mode", choices=["trades", "bars"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--retry-empty", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    for key in ("mode", "start", "end"):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    start, end = date.fromisoformat(cfg["start"]), date.fromisoformat(cfg["end"])
    if start > end or end >= datetime.now(ZoneInfo("America/New_York")).date():
        parser.error("Require start <= end and a completed historical date before today in New York.")
    if cfg["requests_per_minute"] <= 0 or cfg["product_snapshot_days"] < 1:
        parser.error("requests_per_minute and product_snapshot_days must be positive")
    if cfg["mode"] not in {"trades", "bars"} or cfg["stock_bar_size"] not in {"minute", "hour", "day"} or cfg["futures_bar_size"] not in {"1sec", "1min", "1hour", "1session"}:
        parser.error("Unsupported mode or bar size; see config.example.json and README")
    root = (config_path.parent / cfg["output"]).resolve()
    archive = Archive(root)
    issues = [f'{e["symbol"]}: {e["note"]}' for e in cfg["equities"] if e.get("note")]
    issues += ["Requested futures history before 2017-04-03 requires custom vendor sourcing.",
               "All global agricultural futures and LME nickel/aluminium are not guaranteed by Massive's CME-group coverage.",
               "Successful requests and empty responses do not establish complete market coverage."]
    atomic_json(root / "scope_issues.json", issues)
    if args.command == "report":
        print(json.dumps(archive.report(), indent=2))
        return
    if args.command == "plan":
        print(json.dumps({"requested_start": str(start), "requested_end_inclusive": str(end),
            "mode": cfg["mode"], "equity_request_chunks": sum(1 for _ in jobs(cfg, {}, "stocks")),
            "futures": "Run discover with a key to enumerate actual products and contracts.", "issues": issues}, indent=2))
        return
    client = Client(os.environ.get("MASSIVE_API_KEY") or cfg["api_key"], cfg["requests_per_minute"])
    if args.command == "discover":
        universe = discover(client, cfg, root)
        print(f'Discovered {len(universe["contracts"])} candidate contracts; inspect products.json and universe.json.')
        return
    universe = {"contracts": []}
    if args.asset != "stocks":
        universe_path = root / "universe.json"
        if universe_path.exists():
            universe = json.loads(universe_path.read_text())
            if universe.get("config_fingerprint") != discovery_fingerprint(cfg):
                raise ValueError("Discovery settings changed. Run discover again for this configuration.")
        else:
            universe = discover(client, cfg, root)
    errors = 0
    blocked = set()
    try:
        for spec in jobs(cfg, universe, args.asset):
            series = (spec["asset"], spec["symbol"])
            if series in blocked:
                archive.save(spec, "blocked", error="Earlier HTTP 404 in this series; review symbol and rerun")
                continue
            try:
                archive.download(client, spec, args.retry_empty)
            except APIError as exc:
                errors += 1
                print(f'{spec["symbol"]}: {exc}', file=sys.stderr)
                if exc.status in {401, 429}:
                    raise
                if exc.status == 404:
                    blocked.add(series)
    finally:
        summary = archive.report()
        print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (APIError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
