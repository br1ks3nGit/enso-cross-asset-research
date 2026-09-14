"""Data and numerical checks for the delivered scientific analysis."""
import json, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from prepare import oni, temperatures
from model import load_sst, fit, ROOT, RAW, OUT

class ForecastTests(unittest.TestCase):
    def test_oni_definition_cutoff_and_no_duplicates(self):
        d=oni();self.assertFalse(d.duplicated(['year','month']).any())
        latest=d.iloc[-1]
        self.assertEqual((latest.year,latest.month,latest.season),(2026,7,'JJA'))
        self.assertEqual(latest.oni_c,1.8)
        self.assertEqual(d[d.month==12].season.unique().tolist(),['NDJ'])
        self.assertFalse(((d.year==2026)&(d.month>=8)).any())

    def test_noaa_state_conversion_and_missing_data(self):
        d=temperatures();self.assertEqual(d.state.nunique(),50)
        r=d[(d.state=='Alabama')&(d.year==1895)&(d.month==1)].iloc[0]
        self.assertAlmostEqual(r.temperature_c,(43.1-32)/1.8)
        self.assertFalse(((d.year==2026)&(d.month>8)).any())
        self.assertEqual(d[d.state=='Hawaii'].year.min(),1991)

    def test_grid_mean_and_forecast_alignment(self):
        ds=xr.open_dataset(RAW/'ecmwf_51_forecast_sst.nc')
        self.assertEqual(ds.sst.attrs['units'],'K')
        self.assertEqual(ds.forecastMonth.values.tolist(),[1,2,3,4,5])
        self.assertEqual(str(ds.forecast_reference_time.values[0])[:10],'2026-09-01')
        a=ds.sst.values[0,0,0]
        w=np.broadcast_to(np.cos(np.deg2rad(ds.latitude.values))[:,None],a.shape)
        expected=float(np.sum(a*w)/np.sum(w)-273.15)
        self.assertAlmostEqual(float(load_sst('ecmwf','forecast').values[0,0,0]),expected,places=5)

    def test_historical_prediction_reconstructed_without_future_targets(self):
        bundle=json.loads((OUT/'oni_forecast.json').read_text())
        years=np.array(bundle['training_years']);self.assertNotIn(2024,years)
        d=load_sst('cmcc','hindcast').sel(forecast_reference_time=[f'{y}-09-01' for y in years])
        obs=oni().pivot(index='year',columns='month',values='oni_c')
        xx=np.column_stack([np.nanmean(d.values[:,:,[1,2,3]],axis=(1,2)),obs.loc[years,7]])
        mask=years<2020
        pred=fit(xx[mask],obs.loc[years[mask],11].values,.1)(xx[years==2020])[0]
        saved=next(r for r in bundle['verification'] if r['year']==2020 and r['month']==11)
        self.assertAlmostEqual(pred,saved['cmcc_prediction_c'],places=9)

    def test_effects_are_neutral_counterfactual_differences(self):
        o=json.loads((OUT/'oni_forecast.json').read_text());f={r['month']:r['oni_c'] for r in o['forecast']}
        s=json.loads((OUT/'state_enso_effects.json').read_text())['estimates']
        self.assertEqual(len(s),200);self.assertEqual(len({(r['state'],r['month']) for r in s}),200)
        for r in s:
            self.assertAlmostEqual(r['enso_effect_c'],r['coefficient_c_per_oni_c']*f[r['month']],places=9)
            self.assertAlmostEqual(r['enso_effect_f'],r['enso_effect_c']*1.8,places=9)
            self.assertLess(r['p10_c'],r['p90_c'])
            self.assertGreater(r['residual_weather_sd_c'],0)
            self.assertLessEqual(r['training_end'],2025)

    def test_outputs_are_finite_and_intervals_are_ordered(self):
        o=json.loads((OUT/'oni_forecast.json').read_text())
        for r in o['forecast']:
            self.assertLess(r['p10_c'],r['oni_c']);self.assertLess(r['oni_c'],r['p90_c'])
        self.assertTrue(np.isfinite(np.load(OUT/'oni_draws.npy')).all())
        self.assertEqual(len(o['verification']),32)

if __name__=='__main__':unittest.main()
