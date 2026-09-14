import unittest

import pandas as pd

import run_analysis as a


class WalkForwardAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.equities = a.load_inputs()

    def test_publication_clock(self):
        candidates = a.oni_candidates(self.base, publication_delay=2)
        target = pd.Period("2020-06", "M")
        self.assertEqual(candidates["ONI lag 0"].loc[target], self.base["ONI"].loc[pd.Period("2020-03", "M")])
        self.assertEqual(candidates["dONI lag 2"].loc[target], self.base["ONI"].diff().loc[pd.Period("2020-01", "M")])

    def test_every_forecast_is_after_training_cutoff(self):
        wf = a.walk_forward(self.base["r_Copper"], a.oni_candidates(self.base, 2), min_train=120)
        self.assertTrue((pd.PeriodIndex(wf["month"], freq="M") > pd.PeriodIndex(wf["train_end"], freq="M")).all())

    def test_specification_is_frozen_inside_each_block(self):
        wf = a.walk_forward(self.base["r_Copper"], a.oni_candidates(self.base, 2), min_train=120)
        self.assertTrue((wf.groupby("train_end")["spec"].nunique() == 1).all())
        self.assertTrue((wf.groupby("train_end")["alpha"].nunique() == 1).all())
        self.assertTrue((wf.groupby("train_end")["beta"].nunique() == 1).all())

    def test_partial_september_bar_is_excluded(self):
        self.assertLessEqual(self.equities["SPY"].dropna().index.max(), a.END_MONTH)


if __name__ == "__main__":
    unittest.main()
