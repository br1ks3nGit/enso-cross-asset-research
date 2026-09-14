#!/usr/bin/env python3

import unittest

import run_analysis as analysis


class CoolingCapexResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = analysis.load_states()
        cls.prices, cls.price_stats = analysis.load_prices()

    def test_selected_states_and_winter_temperature_screen(self):
        self.assertEqual(len(self.states), 11)
        self.assertEqual(set(self.states["month"]), {12})
        self.assertLess(self.states["implied_dec_p90_mean_c"].max(), 5.0)

    def test_no_design_breach_means_zero_central_weather_capex(self):
        disclosed = self.states[self.states["capacity_2025_gw"].notna()]
        self.assertEqual(disclosed["weather_driven_capex_central_usd"].sum(), 0.0)
        self.assertAlmostEqual(disclosed["audit_screen_usd_central"].sum(), 5_803_707.317073171, places=2)

    def test_capacity_pipeline(self):
        disclosed = self.states[self.states["capacity_2025_gw"].notna()]
        self.assertAlmostEqual(disclosed["capacity_2025_gw"].sum(), 4.576, places=3)
        self.assertAlmostEqual(disclosed["pipeline_increment_gw"].sum(), 23.643, places=3)

    def test_massive_cutoff_and_supplier_coverage(self):
        wanted = {row["ticker"] for row in analysis.SUPPLIERS}
        available = set(self.price_stats["ticker"])
        self.assertTrue(wanted.issubset(available))
        self.assertEqual(set(self.price_stats["last_date"]), {"2026-09-11"})

    def test_report_contains_required_distinctions(self):
        report = analysis.make_report(self.states, self.prices, self.price_stats)
        self.assertIn("new cooling equipment capex attributable only to this ENSO forecast is $0", report)
        self.assertIn("operating expense, not equipment capex", report)
        self.assertIn("Vendor-matching limitation", report)
        self.assertIn("<h2>Sources</h2>", report)


if __name__ == "__main__":
    unittest.main()
