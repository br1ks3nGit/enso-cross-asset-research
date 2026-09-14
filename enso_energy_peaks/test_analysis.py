import unittest

import numpy as np
import pandas as pd

import run_analysis as a


class EnergyPeakAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oni = a.load_oni()

    def test_information_delay(self):
        asof = self.oni.shift(2)
        self.assertEqual(asof.loc[pd.Period("2020-06", "M")], self.oni.loc[pd.Period("2020-04", "M")])

    def test_peak_confirmation_timing(self):
        peaks = a.detect_peaks(self.oni)
        row = peaks.loc[peaks["peak_month"] == "2015-12"].iloc[0]
        decline = pd.Period(row["first_decline_center_month"], "M")
        decision = pd.Period(row["confirmation_decision_month"], "M")
        trade = pd.Period(row["trade_start_month"], "M")
        self.assertEqual(decision, decline + 2)
        self.assertEqual(trade, decision + 1)

    def test_total_return_series_is_finite_and_complete_month_only(self):
        for ticker in a.ALL:
            price = a.load_total_return_prices(ticker)
            self.assertTrue(np.isfinite(price).all())
            self.assertGreater((price > 0).mean(), 0.99)
            self.assertLessEqual(price.index.max(), a.END)

    def test_corporate_action_floors(self):
        self.assertGreaterEqual(a.load_total_return_prices("COP").index.min(), pd.Period("2012-06", "M"))
        self.assertGreaterEqual(a.load_total_return_prices("MPC").index.min(), pd.Period("2011-07", "M"))
        self.assertGreaterEqual(a.load_total_return_prices("PSX").index.min(), pd.Period("2012-05", "M"))


if __name__ == "__main__":
    unittest.main()
