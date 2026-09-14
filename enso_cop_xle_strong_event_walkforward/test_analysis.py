import json
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calculated"
sys.path.insert(0, str(ROOT))
import run_analysis as analysis


class Parser(HTMLParser):
    pass


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = pd.read_csv(OUT / "walkforward_signal_observations.csv", parse_dates=["entry_date", "exit_date"])
        cls.trades = pd.read_csv(OUT / "non_overlapping_trades.csv", parse_dates=["entry_date", "exit_date"])

    def test_massive_daily_coverage_and_dividend_fields(self):
        for ticker in ("COP", "XLE"):
            bars = json.loads((ROOT / "raw" / f"massive_{ticker}_daily.json").read_text())
            self.assertEqual(len(bars["results"]), 3590)
            dividends = json.loads((ROOT / "raw" / f"massive_{ticker}_dividends.json").read_text())["results"]
            self.assertTrue(all("split_adjusted_cash_amount" in row for row in dividends))

    def test_signal_timing_and_positive_oni_only(self):
        test = self.obs[self.obs["is_test"]]
        self.assertTrue((test["ONI"] >= 1.0).all())
        centered = pd.PeriodIndex(test["centered_month"], freq="M")
        information = pd.PeriodIndex(test["information_month"], freq="M")
        self.assertTrue((information == centered + 2).all())
        cutoff = information.to_timestamp(how="end").normalize()
        self.assertTrue((test["entry_date"].to_numpy() > cutoff.to_numpy()).all())

    def test_walkforward_training_is_purged(self):
        for _, row in self.obs[self.obs["direction"].notna()].iterrows():
            same = self.obs[(self.obs["rule"] == row["rule"]) & (self.obs["horizon_days"] == row["horizon_days"])]
            prior = same[(same["exit_date"] < row["entry_date"]) & same["spread_return"].notna()]
            self.assertEqual(int(row["train_n"]), len(prior))

    def test_non_overlapping_and_cost(self):
        self.assertTrue((self.trades["commission_return"] == analysis.PAIR_ROUND_TRIP_COST).all())
        for _, group in self.trades.groupby(["rule", "horizon_days"]):
            group = group.sort_values("entry_date")
            self.assertTrue((group["entry_date"].iloc[1:].to_numpy() > group["exit_date"].iloc[:-1].to_numpy()).all())

    def test_headline_reconciles(self):
        summary = pd.read_csv(OUT / "strategy_summary.csv")
        best = summary.iloc[0]
        self.assertEqual(best["rule"], "S5 Strong but cooling")
        self.assertEqual(int(best["horizon_days"]), 40)
        self.assertAlmostEqual(float(best["after_tax_cumulative_return"]), 0.030832, places=6)
        benchmark = pd.read_csv(OUT / "energy_benchmark.csv")
        self.assertAlmostEqual(float(benchmark.iloc[0]["after_tax_return"]), 1.453186, places=6)

    def test_report_is_self_contained_and_has_no_key(self):
        report = (ROOT / "COP_XLE_strong_El_Nino_walkforward_report.html").read_text()
        Parser().feed(report)
        key = json.loads((ROOT.parent / "config.json").read_text())["api_key"]
        if key:
            self.assertNotIn(key, report)
        for phrase in ("Signal quality", "Market regimes", "Energy benchmark and tax sensitivity", "Walk-forward mechanics"):
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
