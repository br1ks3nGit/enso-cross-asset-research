import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calculated"


class Parser(HTMLParser):
    pass


class OptionBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selections = pd.read_csv(ROOT / "contract_selections.csv", parse_dates=["entry_date", "planned_exit_date", "expiration_date"])
        cls.components = pd.read_csv(OUT / "option_component_prices.csv")
        cls.trades = pd.read_csv(OUT / "option_strategy_trades.csv")
        cls.summary = pd.read_csv(OUT / "option_strategy_summary.csv")

    def test_contracts_are_point_in_time_and_exact_bar_dates_exist(self):
        self.assertEqual(len(self.selections), 8)
        self.assertTrue((self.selections["expiration_date"] >= self.selections["planned_exit_date"] + pd.Timedelta(days=7)).all())
        for _, row in self.selections.iterrows():
            payload = json.loads((ROOT / "raw" / f"bars_{row['contract'].replace(':', '_')}.json").read_text())
            dates = {pd.to_datetime(x["t"], unit="ms", utc=True).tz_convert(None).normalize() for x in payload["results"]}
            self.assertIn(row["entry_date"], dates)
            self.assertIn(row["planned_exit_date"], dates)

    def test_only_positive_strong_but_cooling_signals(self):
        self.assertTrue((self.trades["ONI"] >= 1.0).all())
        self.assertTrue((self.trades["dONI"] < 0).all())
        self.assertEqual(set(self.trades["centered_month"]), {"2024-01", "2024-03"})

    def test_debit_budget_and_reconciliation(self):
        self.assertTrue((self.trades["capital_deployed"] <= self.trades["portfolio_before"] * 0.10 + 1e-9).all())
        for _, summary in self.summary.iterrows():
            detail = self.trades[self.trades["structure"] == summary["structure"]]
            self.assertAlmostEqual(detail["net_pre_tax_pnl"].sum(), summary["net_pre_tax_profit"], places=8)
            self.assertAlmostEqual(detail["friction_cost"].sum(), summary["friction_cost"], places=8)
            expected_tax = 0.37 * max(summary["net_pre_tax_profit"], 0)
            self.assertAlmostEqual(expected_tax, summary["tax_paid"], places=8)

    def test_headline_result(self):
        best = self.summary.iloc[0]
        self.assertEqual(best["structure"], "O2 Long COP ATM put")
        self.assertAlmostEqual(best["after_tax_return"], 0.0834442875, places=8)
        self.assertEqual(int(best["trades"]), 2)

    def test_report_has_no_credential(self):
        report = (ROOT / "ONI_COP_XLE_options_adaptation_report.html").read_text()
        Parser().feed(report)
        key = json.loads((ROOT.parent / "config.json").read_text())["api_key"]
        if key:
            self.assertNotIn(key, report)
        for phrase in ("Five option structures", "Contract selection", "Execution and tax assumptions", "Limitations"):
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
