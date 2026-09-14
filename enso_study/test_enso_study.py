import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
import enso_study_analyze as s


class StudyTests(unittest.TestCase):
    def test_weekly_fixed_width_negative_fields(self):
        weekly,oni=s.parse_climate()
        self.assertEqual(weekly.iloc[0].nino34,-.2)
        djf=oni[oni.center_month=='2024-01'].iloc[0]
        self.assertEqual(djf.observation_end,pd.Timestamp('2024-02-29'))
        self.assertEqual(djf.available_assumption,pd.Timestamp('2024-03-15'))

    def test_signal_cannot_use_future_week(self):
        weekly=pd.DataFrame({'center_date':pd.to_datetime(['2024-01-10','2024-01-24']),'nino34':[-1.,2.]})
        oni=pd.DataFrame({'available_assumption':pd.to_datetime(['2023-12-15']),'oni':[-.5]})
        feature=s.climate_features(pd.period_range('2024-02','2024-02',freq='M'),pd.to_datetime(['2024-01-31']),weekly,oni,14)
        self.assertEqual(feature.iloc[0].enso,-1.)
        self.assertEqual(feature.iloc[0].phase,'Cold')

    def test_dividend_and_split_consistency(self):
        bars=[{'t':int(pd.Timestamp('2024-01-02',tz='UTC').timestamp()*1000),'c':10.,'v':100},
              {'t':int(pd.Timestamp('2024-01-03',tz='UTC').timestamp()*1000),'c':9.5,'v':100}]
        # UTC-midnight bars map to previous New York date in this synthetic fixture.
        fixtures={'equity_X':{'results':bars},'dividends_X':{'results':[{'id':'d','ex_dividend_date':'2024-01-02','cash_amount':1.}]},
                  'splits_X':{'results':[{'id':'s','execution_date':'2024-06-01','split_from':1,'split_to':2}]}}
        with patch.object(s,'load',side_effect=lambda k:fixtures[k]):
            d=s.stock('X')
        self.assertAlmostEqual(d.iloc[-1].price_return,-.05)
        self.assertAlmostEqual(d.iloc[-1].total_return,0.)

    def test_no_forward_fill_across_large_gap(self):
        close=pd.Series([10.,20.,30.],index=pd.to_datetime(['2024-01-01','2024-01-03','2024-02-01']))
        ret=s.no_gap_returns(close)
        self.assertAlmostEqual(ret.iloc[1],1.)
        self.assertTrue(np.isnan(ret.iloc[2]))

    def test_monthly_compounding_and_partial_exclusion(self):
        dates=pd.bdate_range('2024-01-01','2024-03-08')
        d=pd.DataFrame({'close':100*1.001**np.arange(len(dates)),'price_return':.001,'total_return':.002},index=dates)
        m=s.monthly(d)
        self.assertNotIn(pd.Period('2024-03'),m.index)
        self.assertAlmostEqual(m.loc[pd.Period('2024-02'),'total_return'],1.002**21-1)

    def test_known_correlation_and_controls(self):
        self.assertAlmostEqual(s.corr([1,2,3,4,5],[10,8,6,4,2]),-1.)
        self.assertTrue(np.isnan(s.corr([1]*5,[1,2,3,4,5])))

    def test_costs_and_prior_exposure(self):
        months=pd.period_range('2017-01',periods=2,freq='M')
        returns=pd.DataFrame({'A':[.1,.2]},index=months)
        features=pd.DataFrame({'enso':[1.5,0.],'enso_abs':[1.5,0.],'hurricane_season':[False,False]},index=months)
        result=s.strategy_test(returns,features,'agriculture')['models']['ENSO overlay']
        first=(1-.001*.5)*1.05-1
        before=.5*1.1/1.05
        second=(1-.001*(1-before))*1.2-1
        self.assertAlmostEqual(result['returns'][0],first)
        self.assertAlmostEqual(result['returns'][1],second)
        self.assertAlmostEqual(result['total_return'],(1+first)*(1+second)-1)

    def test_probability_forecast_uses_only_prior_outcomes(self):
        months=pd.period_range('2010-01','2017-03',freq='M')
        frame=pd.DataFrame({'price_return':np.tile([-.1,.1],len(months))[:len(months)],'phase':'Cold'},index=months)
        original=s.calibration(frame)
        changed=frame.copy();changed.loc[pd.Period('2017-03'),'price_return']=-9.
        updated=s.calibration(changed)
        self.assertEqual(original['predictions'][-1]['p_enso'],updated['predictions'][-1]['p_enso'])

    def test_downloaded_data_and_result_integrity(self):
        r=json.loads((s.CALC/'results.json').read_text())
        coverage={x['asset']:x for x in r['coverage']}
        self.assertEqual(len([x for x in coverage if x in s.FUTURES]),16)
        for name in s.INSURERS:
            self.assertEqual(coverage[name]['daily_rows'],5030)
        for p in r['downside_probabilities']:
            self.assertAlmostEqual(p['probability'],p['negative_months']/p['months'])
            self.assertTrue(0<=p['ci_low']<=p['ci_high']<=1)
        for test in r['backtests']:
            for model in test['models'].values():
                self.assertEqual(model['months'],116)
                self.assertAlmostEqual(np.prod(1+np.array(model['returns']))-1,model['total_return'])
                self.assertTrue(model['max_drawdown']<=0)
        for symbol in s.FUTURES:
            d=pd.read_json(s.CALC/('daily_'+symbol+'.jsonl'),lines=True)
            self.assertFalse(pd.to_datetime(d.date).duplicated().any())


if __name__=='__main__':unittest.main()
