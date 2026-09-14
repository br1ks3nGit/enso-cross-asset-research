import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calculated"


class Parser(HTMLParser):
    pass


class StraddleBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selections = pd.read_csv(ROOT / "contract_selections.csv", parse_dates=["entry_date", "exit_date", "expiration_date"])
        cls.metrics = pd.read_csv(OUT / "straddle_trade_metrics.csv")
        cls.sim = pd.read_csv(OUT / "capital_constrained_trades.csv")
        cls.summary = pd.read_csv(OUT / "strategy_summary.csv")

    def test_contract_selection_constraints(self):
        self.assertTrue(self.selections["dte_at_entry"].between(45, 70).all())
        self.assertTrue((self.selections["strike"] > 0).all())
        self.assertTrue((self.selections[["call_entry_volume", "put_entry_volume", "call_exit_volume", "put_exit_volume"]] > 0).all().all())

    def test_hkd_cash_and_friction_reconcile(self):
        expected = self.metrics["exit_proceeds_hkd"] - self.metrics["entry_debit_hkd"]
        self.assertTrue(((expected - self.metrics["net_pnl_hkd"]).abs() < 1e-7).all())
        friction = self.metrics["raw_pnl_hkd"] - self.metrics["net_pnl_hkd"]
        self.assertTrue(((friction - self.metrics["friction_cost_hkd"]).abs() < 1e-7).all())
        self.assertTrue((self.metrics["friction_cost_hkd"] >= -1e-7).all())

    def test_no_fractional_or_leveraged_execution(self):
        executed = self.sim[self.sim["status"] == "Executed: one straddle"]
        self.assertTrue((executed["entry_debit_hkd"] <= executed["equity_before_hkd"] + 1e-7).all())
        skipped = self.sim[self.sim["status"].str.startswith("Skipped")]
        self.assertTrue((skipped["realized_pnl_hkd"] == 0).all())

    def test_oni_delay_and_gate(self):
        signals = pd.read_csv(OUT / "oni_signal_inputs.csv")
        self.assertEqual(signals.loc[signals["decision_month"] == "2026-04", "available_centered_month"].iloc[0], "2026-01")
        self.assertEqual(signals.loc[signals["decision_month"] == "2026-09", "available_centered_month"].iloc[0], "2026-06")
        self.assertEqual(set(signals.loc[signals["oni_gate"], "decision_month"]), {"2026-07", "2026-08", "2026-09"})

    def test_report_is_valid_and_has_no_api_key(self):
        report = (ROOT / "DC_infrastructure_straddles_Apr_Sep_2026.html").read_text()
        Parser().feed(report)
        key = json.loads((ROOT.parent / "config.json").read_text())["api_key"]
        if key:
            self.assertNotIn(key, report)
        for phrase in ("Capital-constrained results", "Signal quality, separate from P&amp;L", "Exact formulas", "Missing selections"):
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
