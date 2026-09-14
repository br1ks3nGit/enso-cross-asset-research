import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from datetime import date

import massive_history as h
import climate_join as cj


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parent.parent / "config.example.json").read_text())

    def test_calendar_chunks_have_no_gaps(self):
        parts = list(h.chunks(date(2020, 2, 28), date(2020, 3, 3), 2))
        self.assertEqual(parts, [(date(2020, 2, 28), date(2020, 3, 1)), (date(2020, 3, 1), date(2020, 3, 3))])

    def test_eastern_day_dst(self):
        day_ns = h.ns_at(date(2024, 3, 11), "America/New_York") - h.ns_at(date(2024, 3, 10), "America/New_York")
        self.assertEqual(day_ns, 23 * 3600 * 10**9)

    def test_nanoseconds_are_exact(self):
        n = 1734472799000509201
        row = h.normalize({"timestamp": n, "price": 10}, "futures", "trades", {})
        self.assertEqual(row["timestamp_ns"], n)
        self.assertTrue(row["timestamp_utc"].endswith(".000509201Z"))
        stock = h.normalize({"t": 1734472799000, "c": 12}, "stocks", "bars", {})
        self.assertEqual(stock["timestamp_ns"], 1734472799000000000)

    def test_pagination_keeps_next_cursor_only(self):
        client = h.Client("test")
        with patch.object(client, "get", side_effect=[{"results": [1], "next_url": h.BASE + "/next?cursor=x"}, {"results": [2]}]) as get:
            self.assertEqual(list(client.pages("/first", {"limit": 1})), [[1], [2]])
            self.assertEqual(get.call_args_list[1].args, (h.BASE + "/next?cursor=x", None))

    def test_pagination_loop_fails(self):
        client = h.Client("test")
        with patch.object(client, "get", return_value={"results": [], "next_url": "/same"}):
            with self.assertRaises(h.APIError):
                list(client.pages("/same"))

    def test_keys_removed_and_foreign_hosts_rejected(self):
        self.assertEqual(h.clean_url("/next?cursor=abc&apiKey=REDACTED"), h.BASE + "/next?cursor=abc")
        with self.assertRaises(h.APIError):
            h.clean_url("https://untrusted.example/next")

    def test_aliases_do_not_cross_transaction_boundary(self):
        self.cfg.update(start="2020-03-01", end="2020-03-03")
        jobs = [j for j in h.jobs(self.cfg, {}, "stocks") if j["instrument"]["symbol"] == "TT"]
        self.assertEqual([(j["symbol"], j["start"]) for j in jobs], [("IR", "2020-03-01"), ("TT", "2020-03-02"), ("TT", "2020-03-03")])
        self.assertFalse(any(j["symbol"] == "AGCL" for j in h.jobs(self.cfg, {}, "stocks")))

    def test_agriculture_and_metals_discovery(self):
        self.assertTrue(h.product_reason({"name": "Soybean Oil", "type": "single"}, self.cfg))
        self.assertTrue(h.product_reason({"name": "Aluminium"}, self.cfg))
        self.assertIsNone(h.product_reason({"name": "Copper", "type": "combo"}, self.cfg))
        self.assertIsNone(h.product_reason({"name": "Natural Gas"}, self.cfg))

    def test_session_captures_previous_evening(self):
        self.cfg.update(start="2024-01-03", end="2024-01-03", mode="bars", futures_bar_size="1session")
        universe = {"contracts": [{"contract": {"ticker": "TEST", "first_trade_date": "2024-01-03", "last_trade_date": "2024-02-01"}, "product": {}}]}
        job = next(h.jobs(self.cfg, universe, "futures"))
        self.assertEqual(job["start"], "2024-01-02")
        self.assertEqual(job["session_start"], "2024-01-03")

    def test_atomic_resume_and_equal_time_distinct_trades(self):
        spec = h.request_spec(self.cfg, "stocks", "MOD", {}, date(2024, 1, 2), date(2024, 1, 3))
        ts = spec["lower_ns"]
        class Fake:
            calls = 0
            def pages(self, *args):
                self.calls += 1
                yield [{"sip_timestamp": ts, "id": "1"}, {"sip_timestamp": ts, "id": "2"}, {"sip_timestamp": spec["upper_ns"], "id": "3"}]
        with tempfile.TemporaryDirectory() as tmp:
            archive, client = h.Archive(tmp), Fake()
            archive.download(client, spec)
            archive.download(client, spec)
            self.assertEqual(client.calls, 1)
            self.assertEqual(archive.report(), {"complete": 1})
            path = next(Path(tmp).rglob("*.gz"))
            with gzip.open(path, "rt") as stream:
                self.assertEqual(len(list(stream)), 2)
            archive.db.close()

    def test_failed_pages_not_marked_complete(self):
        spec = h.request_spec(self.cfg, "stocks", "MOD", {}, date(2024, 1, 2), date(2024, 1, 3))
        class Broken:
            def pages(self, *args):
                yield [{"sip_timestamp": spec["lower_ns"]}]
                raise h.APIError("simulated disconnect")
        with tempfile.TemporaryDirectory() as tmp:
            archive = h.Archive(tmp)
            with self.assertRaises(h.APIError):
                archive.download(Broken(), spec)
            self.assertEqual(archive.report(), {"error": 1})
            self.assertFalse(list(Path(tmp).rglob("*.gz")))
            archive.db.close()

    def test_climate_release_and_revision_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "features.jsonl"
            records = [
                {"scope": "global", "feature": "oni", "observed_until": "2024-03-01T00:00:00Z", "available_at": "2024-03-05T00:00:00Z", "value": 1.0, "point_in_time": True},
                {"scope": "global", "feature": "oni", "observed_until": "2024-03-01T00:00:00Z", "available_at": "2024-04-05T00:00:00Z", "value": 1.1, "point_in_time": True}]
            file.write_text("\n".join(json.dumps(r) for r in records))
            db = sqlite3.connect(":memory:")
            cj.index_features(db, file)
            self.assertIsNone(cj.asof(db, "global", "oni", "2024-03-04T00:00:00Z", 120))
            self.assertEqual(cj.asof(db, "global", "oni", "2024-03-06T00:00:00Z", 120)["value"], 1.0)
            self.assertEqual(cj.asof(db, "global", "oni", "2024-04-06T00:00:00Z", 120)["value"], 1.1)
            self.assertIsNone(cj.asof(db, "global", "oni", "2025-04-06T00:00:00Z", 120))
            db.close()


if __name__ == "__main__":
    unittest.main()
