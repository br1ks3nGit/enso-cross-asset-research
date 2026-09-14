"""Seasonal ONI calibration and monthly state ENSO response estimates."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t
from prepare import oni, temperatures, SEASONS

ROOT=Path(__file__).resolve().parent; RAW=ROOT/'raw'; OUT=ROOT/'calculated'
MODELS={'ecmwf':'51','dwd':'22','cmcc':'4'}
MONTHS=[9,10,11,12]
RNG=np.random.default_rng(20260911)

def fit(x,y,alpha=1):
    x=np.asarray(x,float); y=np.asarray(y,float)
    mu=x.mean(0); sd=x.std(0);sd=np.where(sd>1e-8,sd,1)
    z=(x-mu)/sd
    b=np.linalg.solve(z.T@z+np.eye(x.shape[1])*alpha,z.T@(y-y.mean()))
    return lambda new: y.mean()+(np.asarray(new)-mu)/sd@b

def load_sst(model,stage):
    with xr.open_dataset(RAW/f'{model}_{MODELS[model]}_{stage}_sst.nc') as ds:
        d=ds.sst.weighted(np.cos(np.deg2rad(ds.latitude))).mean(['latitude','longitude']).load()
    assert d.attrs.get('units','K') in ['K','kelvin']
    return d.transpose('forecast_reference_time','number','forecastMonth')-273.15

def forecast():
    obs=oni().pivot(index='year',columns='month',values='oni_c')
    ds={m:load_sst(m,'hindcast') for m in MODELS}
    available={m:set(pd.DatetimeIndex(d.forecast_reference_time.values).year) for m,d in ds.items()}
    years=np.array(sorted(set.intersection(*available.values())))
    init=obs.loc[years,7].values
    for m in ds:
        ds[m]=ds[m].sel(forecast_reference_time=[f'{y}-09-01' for y in years])
    fs={m:load_sst(m,'forecast') for m in MODELS}
    monthly=[]
    for m in MODELS:
        assert np.array_equal(pd.DatetimeIndex(ds[m].forecast_reference_time.values).year,years)
        assert np.array_equal(ds[m].forecastMonth.values,np.arange(1,6))
        for lead in range(5):
            vals=fs[m].values[0,:,lead]; vals=vals[np.isfinite(vals)]
            monthly.append({'model':m,'valid_month':str(pd.Period('2026-09')+lead),'members':len(vals),'sst_mean_c':float(vals.mean()),'sst_p10_c':float(np.quantile(vals,.1)),'sst_p90_c':float(np.quantile(vals,.9))})
    X={}; XF={}; Y={}
    for j,month in enumerate(MONTHS):
        leads=[0,1] if j==0 else list(range(j-1,j+2))
        Y[month]=obs.loc[years,month].values
        X[month]={m:np.column_stack([np.nanmean(ds[m].values[:,:,leads],axis=(1,2)),init]) for m in MODELS}
        XF[month]={m:np.column_stack([np.nanmean(fs[m].values[0][:,leads],axis=1),np.repeat(obs.loc[2026,7],fs[m].sizes['number'])]) for m in MODELS}
        X[month]['observations']=init[:,None]
        XF[month]['observations']=np.array([[obs.loc[2026,7]]])
    # Tune only on 2003–2016. Freeze specification before independent 2017–2025 evaluation.
    tuning=[]
    for alpha in [.1,1.,10.]:
        errors={m:[] for m in [*MODELS,'observations','multi']}
        for year in range(2003,2017):
            if year not in years:continue
            train=years<year; test=years==year
            for month in MONTHS:
                pred={m:float(fit(X[month][m][train],Y[month][train],alpha)(X[month][m][test])[0]) for m in [*MODELS,'observations']}
                pred['multi']=np.mean([pred[m] for m in MODELS])
                for m,p in pred.items():errors[m].append((p-Y[month][test][0])**2)
        for m,v in errors.items():tuning.append({'model':m,'alpha':alpha,'rmse_c':float(np.sqrt(np.mean(v)))})
    # Use dynamics only if they outperform the observation-only baseline in the tuning interval.
    best=min(tuning,key=lambda x:x['rmse_c']);chosen=best['model'];alpha=best['alpha']
    verification=[]; residuals=[]
    for year in range(2017,2026):
        if year not in years:continue
        train=years<year;test=years==year;row=[]
        for month in MONTHS:
            pred={m:float(fit(X[month][m][train],Y[month][train],alpha)(X[month][m][test])[0]) for m in [*MODELS,'observations']}
            pred['multi']=float(np.mean([pred[m] for m in MODELS]));pred['persistence']=float(init[test][0])
            actual=float(Y[month][test][0]);row.append(actual-pred[chosen])
            verification.append({'year':year,'month':month,'season':SEASONS[month-1],'actual_oni_c':actual,**{m+'_prediction_c':p for m,p in pred.items()}})
        residuals.append(row)
    residuals=np.array(residuals)
    result=[];point=[];spread=[];modelpoints=[]
    error_df=len(residuals)-1
    quantile80=float(student_t.ppf(.9,error_df))
    for j,month in enumerate(MONTHS):
        pm={m:fit(X[month][m],Y[month],alpha)(XF[month][m]) for m in [*MODELS,'observations']}
        means={m:float(np.nanmean(v)) for m,v in pm.items()}
        means['multi']=float(np.mean([means[m] for m in MODELS]))
        modelpoints.append({'month':month,**means})
        if chosen=='multi':
            variance=np.mean([np.nanvar(pm[m])+(means[m]-means['multi'])**2 for m in MODELS])
        else:variance=float(np.nanvar(pm[chosen]))
        # Predictive error floor from genuine out-of-sample forecasts; ensemble spread cannot shrink it.
        rmse=float(np.sqrt(np.mean(residuals[:,j]**2)))
        calibration_models=list(MODELS) if chosen=='multi' else [chosen]
        leverages=[]
        for m in calibration_models:
            xx=X[month][m]; mu=xx.mean(0);sd=xx.std(0)
            zz=np.column_stack([np.ones(len(xx)),(xx-mu)/sd])
            zf=np.r_[1,(XF[month][m].mean(0)-mu)/sd]
            leverages.append(float(zf@np.linalg.pinv(zz.T@zz+np.diag([0]+[alpha]*xx.shape[1]))@zf))
        leverage=float(np.mean(leverages))
        model_disagreement=float(np.std([means[m] for m in MODELS]))
        sigma=max(np.sqrt(variance+model_disagreement**2),rmse)*np.sqrt(1+leverage)
        point.append(means[chosen]);spread.append(sigma)
        result.append({'month':month,'season':SEASONS[month-1],'oni_c':means[chosen],'p10_c':means[chosen]-quantile80*sigma,'p90_c':means[chosen]+quantile80*sigma,'verification_rmse_c':rmse,'raw_calibrated_ensemble_sd_c':float(np.sqrt(variance)),'calibration_leverage':leverage,'model_disagreement_sd_c':model_disagreement,'uncertainty_scale_c':float(sigma),'error_df':error_df,'model':chosen,'alpha':alpha,'historical_target_max_c':float(Y[month].max()),'exceeds_training_range':bool(means[chosen]>Y[month].max())})
    # Correlated monthly uncertainty, maintaining cross-season dependence in the validation errors.
    corr=np.corrcoef(residuals.T)*.8+np.eye(4)*.2
    cov=corr*np.outer(spread,spread)
    draws=np.asarray(point)+RNG.multivariate_normal(np.zeros(4),cov,size=3000)/np.sqrt(RNG.chisquare(error_df,size=(3000,1))/error_df)
    np.save(OUT/'oni_draws.npy',draws)
    metrics={m:float(np.sqrt(np.mean([(r[m+'_prediction_c']-r['actual_oni_c'])**2 for r in verification]))) for m in [*MODELS,'observations','multi','persistence']}
    bundle={'forecast':result,'selection':best,'validation_2017_2025_rmse_c':metrics,'tuning_2003_2016':tuning,'model_means':modelpoints,'monthly_raw_sst':monthly,'verification':verification,'training_years':years.tolist(),'missing_years_by_model':{m:sorted(set(range(1993,2026))-v) for m,v in available.items()},'index':'ONI ERSSTv6; centered seasons','uncertainty':'Student-t 80% predictive intervals; scale floor from held-out errors; model disagreement and calibration leverage included; not externally calibrated at record strength'}
    (OUT/'oni_forecast.json').write_text(json.dumps(bundle,indent=2))
    print(json.dumps({k:bundle[k] for k in ['forecast','selection','validation_2017_2025_rmse_c']},indent=2),flush=True)
    return draws

def response_fit(year,x,y,alpha):
    design=np.column_stack([np.ones(len(year)),(np.asarray(year)-1990)/10,np.asarray(x)])
    b=np.linalg.solve(design.T@design+np.diag([0,0,alpha]),design.T@np.asarray(y))
    return b

def state_responses(draws):
    oni_bundle=json.loads((OUT/'oni_forecast.json').read_text())
    point={r['month']:r['oni_c'] for r in oni_bundle['forecast']}
    historical_oni={(r['year'],r['month']):r[oni_bundle['selection']['model']+'_prediction_c'] for r in oni_bundle['verification']}
    d=temperatures().merge(oni()[['year','month','oni_c']],on=['year','month'])
    d=d[(d.year>=1950)&(d.year<=2025)&d.month.isin(MONTHS)]
    rows=[];validation=[];tuning=[]
    for j,month in enumerate(MONTHS):
        groups={s:g.sort_values('year') for s,g in d[d.month==month].groupby('state')}
        errors={a:[] for a in [0.,5.,20.,100.]}
        for state,g in groups.items():
            for year in range(2000,2011):
                tr=g[g.year<year];te=g[g.year==year]
                if len(tr)<30 or te.empty:continue
                for a in errors:
                    b=response_fit(tr.year,tr.oni_c,tr.temperature_c,a)
                    p=float(np.array([1,(year-1990)/10,te.oni_c.iloc[0]])@b)
                    errors[a].append((p-te.temperature_c.iloc[0])**2)
        alpha=min(errors,key=lambda a:np.mean(errors[a]));tuning.append({'month':month,'alpha':alpha,'validation_rmse_c':{str(k):float(np.sqrt(np.mean(v))) for k,v in errors.items()}})
        for state,g in groups.items():
            for year in range(2011,2026):
                tr=g[g.year<year];te=g[g.year==year]
                if len(tr)<20 or te.empty:continue
                b=response_fit(tr.year,tr.oni_c,tr.temperature_c,alpha)
                b0=np.linalg.lstsq(np.column_stack([np.ones(len(tr)),(tr.year-1990)/10]),tr.temperature_c,rcond=None)[0]
                p=float(np.array([1,(year-1990)/10,te.oni_c.iloc[0]])@b)
                p0=float(np.array([1,(year-1990)/10])@b0)
                record={'state':state,'month':month,'year':year,'actual_temperature_c':float(te.temperature_c.iloc[0]),'trend_enso_c':p,'trend_only_c':p0}
                if (year,month) in historical_oni:
                    record['forecast_oni_temperature_c']=float(np.array([1,(year-1990)/10,historical_oni[year,month]])@b)
                validation.append(record)
            b=response_fit(g.year,g.oni_c,g.temperature_c,alpha)
            # Circular moving block resampling of five-year paired observations.
            n=len(g);bs=[]
            gy=g.year.to_numpy();gx=g.oni_c.to_numpy();gt=g.temperature_c.to_numpy()
            for k in range(len(draws)):
                starts=RNG.integers(0,n,size=int(np.ceil(n/5)))
                idx=np.concatenate([(start+np.arange(5))%n for start in starts])[:n]
                bs.append(response_fit(gy[idx],gx[idx],gt[idx],alpha)[2])
            effect=np.asarray(bs)*draws[:,j]
            fitted=np.column_stack([np.ones(n),(g.year-1990)/10,g.oni_c])@b
            v=[r for r in validation if r['state']==state and r['month']==month]
            mse=np.mean([(r['trend_enso_c']-r['actual_temperature_c'])**2 for r in v]);mse0=np.mean([(r['trend_only_c']-r['actual_temperature_c'])**2 for r in v])
            lo,hi=np.quantile(effect,[.1,.9]);center=float(b[2]*point[month])
            rows.append({'state':state,'month':month,'season':SEASONS[month-1],'enso_effect_c':center,'enso_effect_f':center*1.8,'p10_c':float(lo),'p90_c':float(hi),'coefficient_c_per_oni_c':float(b[2]),'coefficient_p025':float(np.quantile(bs,.025)),'coefficient_p975':float(np.quantile(bs,.975)),'residual_weather_sd_c':float(np.std(g.temperature_c-fitted,ddof=3)),'training_start':int(g.year.min()),'training_end':int(g.year.max()),'n_years':n,'ridge_penalty':alpha,'conditional_verification_rmse_c':float(np.sqrt(mse)),'trend_only_rmse_c':float(np.sqrt(mse0)),'conditional_skill_vs_trend':float(1-mse/mse0),'effect_80pct_interval_excludes_zero':bool(lo>0 or hi<0)})
        print('State response complete',month,flush=True)
    metrics={}
    for month in MONTHS:
        v=[r for r in validation if r['month']==month]
        a=np.mean([(r['trend_enso_c']-r['actual_temperature_c'])**2 for r in v]);b=np.mean([(r['trend_only_c']-r['actual_temperature_c'])**2 for r in v])
        metrics[month]={'conditional_rmse_c':float(np.sqrt(a)),'trend_only_rmse_c':float(np.sqrt(b)),'conditional_skill':float(1-a/b)}
        e=[r for r in v if 'forecast_oni_temperature_c' in r]
        a=np.mean([(r['forecast_oni_temperature_c']-r['actual_temperature_c'])**2 for r in e]);b=np.mean([(r['trend_only_c']-r['actual_temperature_c'])**2 for r in e])
        metrics[month].update({'end_to_end_rmse_c':float(np.sqrt(a)),'end_to_end_trend_baseline_rmse_c':float(np.sqrt(b)),'end_to_end_skill':float(1-a/b),'end_to_end_years':sorted(set(r['year'] for r in e))})
    (OUT/'state_enso_effects.json').write_text(json.dumps({'estimates':rows,'validation':metrics,'tuning':tuning,'interpretation':'ENSO-associated statistical difference from ONI=0 with fitted trend held fixed; not identified causal attribution or full temperature anomaly forecast. Intervals combine coefficient and ONI uncertainty, excluding unrelated weather variability.','validation_note':'2011–2025 chronological conditional evaluation uses observed target-season ONI. This tests the temperature response, not a full forecast issued in September.'},indent=2))
    (OUT/'temperature_validation.json').write_text(json.dumps(validation))
    return rows

def era5_summary():
    with xr.open_dataset(RAW/'era5_pacific_2026.nc') as d:
        rows=[]
        for time in d.valid_time.values:
            sub=d.sel(valid_time=time,latitude=slice(5,-5),longitude=slice(190,240))
            vals={v:float(sub[v].weighted(np.cos(np.deg2rad(sub.latitude))).mean()) for v in ['sst','u10','v10','msl']}
            rows.append({'month':str(pd.Timestamp(time).date()),'nino34_sst_c':vals['sst']-273.15,'nino34_u10_ms':vals['u10'],'nino34_v10_ms':vals['v10'],'nino34_msl_hpa':vals['msl']/100})
    (OUT/'era5_diagnostics.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    draws=forecast();state_responses(draws);era5_summary()
