#!/usr/bin/env python3
"""Join local weather/ONI vintages to explicit forecast decision times.

Inputs and output are JSON Lines, not API calls. Uses an on-disk SQLite index.
Never substitutes observation dates for actual publication/availability times.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import tempfile


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps must have an explicit timezone")
    delta = dt.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def index_features(db, path):
    db.execute("CREATE TABLE features (scope TEXT, feature TEXT, observed INTEGER, available INTEGER, payload TEXT, PRIMARY KEY(scope,feature,observed,available))")
    with Path(path).open() as stream:
        for line in stream:
            r = json.loads(line)
            if r.get("point_in_time") is not True:
                raise ValueError("Require point_in_time=true only for verified historical vintages")
            observed, available = timestamp(r["observed_until"]), timestamp(r["available_at"])
            if available < observed:
                raise ValueError("This loader accepts observations; available_at must be after observation end")
            if not isinstance(r["value"], (int, float)) or isinstance(r["value"], bool) or not math.isfinite(r["value"]):
                raise ValueError("Feature values must be finite numeric values")
            db.execute("INSERT INTO features VALUES (?,?,?,?,?)", (r["scope"], r["feature"], observed, available, json.dumps(r)))
    db.execute("CREATE INDEX feature_asof ON features(scope,feature,observed DESC,available DESC)")
    db.commit()


def asof(db, scope, feature, decision_time, max_age_days):
    cutoff = timestamp(decision_time)
    oldest = cutoff - int(max_age_days * 86400 * 1_000_000)
    result = db.execute("SELECT payload FROM features WHERE scope=? AND feature=? AND available<=? AND observed<=? AND observed>=? ORDER BY observed DESC,available DESC LIMIT 1",
                        (scope, feature, cutoff, cutoff, oldest)).fetchone()
    return json.loads(result[0]) if result else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decisions", required=True, help="JSONL with decision_time, scope; other columns are preserved")
    p.add_argument("--features", required=True, help="JSONL with scope,feature,value,observed_until,available_at,point_in_time")
    p.add_argument("--select", nargs="+", required=True, help="Names such as oni precipitation_30d temperature_anomaly")
    p.add_argument("--max-age-days", type=float, default=120)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.max_age_days <= 0:
        p.error("max-age-days must be positive")
    output = Path(args.output)
    if output.resolve() in {Path(args.decisions).resolve(), Path(args.features).resolve()}:
        p.error("Output must be different from input files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        db = sqlite3.connect(str(Path(tmp) / "features.sqlite"))
        try:
            index_features(db, args.features)
            part = output.with_suffix(output.suffix + ".partial")
            with Path(args.decisions).open() as source, part.open("w") as dest:
                for line in source:
                    row = json.loads(line)
                    row["climate_features"] = {name: asof(db, "global" if name == "oni" else row["scope"], name,
                        row["decision_time"], args.max_age_days) for name in args.select}
                    dest.write(json.dumps(row) + "\n")
            part.replace(output)
        finally:
            db.close()


if __name__ == "__main__":
    main()
