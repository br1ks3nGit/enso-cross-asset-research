import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calculated"


class Parser(HTMLParser):
    pass


class CoolingSpendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = pd.read_csv(OUT / "affected_states_cooling_spend.csv")

    def test_threshold_and_state_set(self):
        self.assertTrue((self.results["enso_effect_c"] > 2.0).all())
        self.assertEqual(set(self.results["month_name"]), {"December"})
        self.assertEqual(set(self.results["state"]), {
            "Illinois", "Indiana", "Iowa", "Michigan", "Minnesota", "Montana",
            "North Dakota", "New York", "Ohio", "South Dakota", "Wisconsin",
        })

    def test_normalized_formula(self):
        row = self.results[self.results["state"] == "Illinois"].iloc[0]
        expected_mwh = (100 * 0.70 / 1.30) * 744 * 0.015 * row["enso_effect_c"]
        self.assertAlmostEqual(row["central_incremental_cooling_mwh_per_100mw"], expected_mwh, places=8)
        expected_spend = expected_mwh * row["commercial_rate_cents_kwh"] * 10
        self.assertAlmostEqual(row["central_incremental_spend_usd_per_100mw"], expected_spend, places=8)

    def test_absolute_estimates_only_for_disclosed_capacity(self):
        disclosed = self.results["capacity_gw_2025_disclosed"].notna()
        absolute = self.results["central_incremental_spend_usd_disclosed_capacity"].notna()
        self.assertTrue((disclosed == absolute).all())
        self.assertEqual(int(disclosed.sum()), 6)

    def test_range_contains_central_total(self):
        x = self.results.dropna(subset=["capacity_gw_2025_disclosed"])
        self.assertLess(x["low_incremental_spend_usd_disclosed_capacity"].sum(), x["central_incremental_spend_usd_disclosed_capacity"].sum())
        self.assertGreater(x["high_incremental_spend_usd_disclosed_capacity"].sum(), x["central_incremental_spend_usd_disclosed_capacity"].sum())

    def test_report_valid_and_reconciled(self):
        report = (ROOT / "ENSO_2026_datacenter_cooling_spend.html").read_text()
        Parser().feed(report)
        self.assertNotIn("nan", report.lower())
        self.assertIn("states above +2°C only", report)
        self.assertIn("not total cooling opex", report)
        provenance = json.loads((OUT / "provenance.json").read_text())
        self.assertEqual(provenance["capacity_policy"], "No absolute state estimate where Table 1 groups the state into Others")


if __name__ == "__main__":
    unittest.main()
