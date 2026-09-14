"""Reproducible, deliberately simple ENSO association study (cached local inputs)."""
from pathlib import Path
import json
import re
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'
CALC=ROOT/'calculated'
CALC.mkdir(exist_ok=True)
INSURERS=['RNR','EG','ACGL']
FUNDS=['DBA','CORN','SOYB','WEAT']
FUTURES={'ZC':'Corn','ZS':'Soybeans','ZW':'Chicago wheat','ZM':'Soybean meal','ZL':'Soybean oil',
         'ZO':'Oats','ZR':'Rough rice','KE':'Kansas wheat','LE':'Live cattle','GF':'Feeder cattle',
         'HE':'Lean hogs','CC':'Cocoa','KC':'Coffee','CT':'Cotton','SB':'Sugar','OJ':'Orange juice'}
PHASES=['Cold','Neutral','Warm']

def load(name):
    return json.loads((RAW/(name+'.json')).read_text())

def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,pd.Period)):return str(x)
    return x

def save(name,data):
    (CALC/name).write_text(json.dumps(clean_json(data),indent=2,allow_nan=False))

def no_gap_returns(close):
    result=close.pct_change(fill_method=None)
    return result.where(close.index.to_series().diff().dt.days<=7)

def stock(symbol):
    aliases=['RE','EG'] if symbol=='EG' else [symbol]
    frames=[]
    for alias in aliases:
        frame=pd.DataFrame(load('equity_'+alias)['results'])
        frame.index=pd.to_datetime(frame.t,unit='ms',utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
        frames.append(frame[['c','v']].rename(columns={'c':'close','v':'volume'}))
    frame=pd.concat(frames).sort_index()
    if frame.index.duplicated().any():raise ValueError('Duplicate equity dates: '+symbol)
    frame['price_return']=no_gap_returns(frame.close)
    dividends=[];splits=[]
    for alias in aliases:
        dividends+=load('dividends_'+alias)['results']
        splits+=load('splits_'+alias)['results']
    dividends=list({r['id']:r for r in dividends}.values())
    splits=list({r.get('id',json.dumps(r,sort_keys=True)):r for r in splits}.values())
    frame['dividend_adjusted']=0.0
    for div in dividends:
        at=pd.Timestamp(div['ex_dividend_date'])
        if at not in frame.index:continue
        factor=1.0
        for sp in splits:
            if pd.Timestamp(sp['execution_date'])>at:
                factor*=sp['split_from']/sp['split_to']
        frame.loc[at,'dividend_adjusted']+=div['cash_amount']*factor
    frame['total_return']=(frame.close+frame.dividend_adjusted)/frame.close.shift(1)-1
    frame['total_return']=frame.total_return.where(frame.index.to_series().diff().dt.days<=7)
    return frame

def futures(code):
    result=load('yahoo_'+code)['chart']['result'][0]
    frame=pd.DataFrame(result['indicators']['quote'][0],index=pd.to_datetime(result['timestamp'],unit='s',utc=True))
    frame.index=frame.index.tz_convert(result['meta'].get('exchangeTimezoneName','America/New_York')).tz_localize(None).normalize()
    input_rows=len(frame)
    weekend=int((frame.index.weekday>=5).sum())
    frame=frame[(frame.index.weekday<5)&frame.close.notna()&(frame.close>0)]
    duplicate=int(frame.index.duplicated().sum())
    frame=frame[~frame.index.duplicated(keep='last')].sort_index()
    frame['price_return']=no_gap_returns(frame.close)
    frame['total_return']=np.nan
    quality={'input_rows':input_rows,'weekend_rows_removed':weekend,'duplicate_dates_removed':duplicate,
             'usable_rows':len(frame),'zero_volume_rows':int((frame.volume==0).sum()),
             'moves_over_10pct':int((frame.price_return.abs()>.10).sum())}
    return frame,quality

def monthly(frame):
    frame=frame.copy()
    frame['vol_return']=frame.total_return if frame.total_return.notna().any() else frame.price_return
    group=frame.groupby(frame.index.to_period('M'))
    result=pd.DataFrame({'price_return':group.close.last().pct_change(fill_method=None),
        'volatility':group.vol_return.apply(lambda s: np.log1p(s.dropna()).std(ddof=1)*np.sqrt(252)),
        'total_return':group.total_return.apply(lambda s:np.prod(1+s.dropna())-1 if len(s.dropna())>=15 else np.nan),
        'observations':group.close.count(),'last_date':group.apply(lambda x:x.index.max())})
    result['first_day']=group.apply(lambda x:x.index.min().day)
    result=result[(result.observations>=15)&(result.last_date.dt.day>=25)&(result.first_day<=7)]
    result.index.name='month'
    return result

def parse_climate():
    weekly=[]
    for line in (RAW/'weekly_nino34.txt').read_text().splitlines():
        m=re.match(r'\s*(\d{2}[A-Z]{3}\d{4})(.*)',line)
        if not m:continue
        values=[float(v) for v in re.findall(r'[-+]?\d+\.\d+',m[2])]
        if len(values)!=8:raise ValueError('Unexpected weekly climate layout')
        weekly.append({'center_date':pd.to_datetime(m[1],format='%d%b%Y'),'nino34':values[5]})
    weekly=pd.DataFrame(weekly).sort_values('center_date')
    seasons=['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']
    oni=[]
    for line in (RAW/'oni.txt').read_text().splitlines()[1:]:
        p=line.split()
        if len(p)!=4:continue
        center=pd.Period(year=int(p[1]),month=seasons.index(p[0])+1,freq='M')
        end=(center+1).end_time.normalize()
        oni.append({'center_month':str(center),'observation_end':end,'available_assumption':end+pd.Timedelta(days=15),'oni':float(p[3])})
    return weekly,pd.DataFrame(oni)

def climate_features(months,market_dates,weekly,oni,lag=14):
    rows=[]
    for month in months:
        prev=month-1
        dates=market_dates[market_dates.to_period('M')==prev]
        if not len(dates):continue
        # Before the previous month's final close; inputs are lagged in whole days.
        decision=dates.max()
        eligible=weekly[weekly.center_date+pd.Timedelta(days=lag)<=decision]
        on=oni[oni.available_assumption<=decision]
        if eligible.empty:continue
        last=eligible.iloc[-1]
        if (decision-last.center_date).days>45:continue
        value=last.nino34
        rows.append({'month':month,'decision_date':decision,'enso_center':last.center_date,'enso':value,
             'enso_abs':abs(value),'phase':'Warm' if value>=.5 else 'Cold' if value<=-.5 else 'Neutral',
             'oni':on.iloc[-1].oni if len(on) else np.nan,'hurricane_season':month.month in range(6,12)})
    return pd.DataFrame(rows).set_index('month')

def corr(x,y):
    pair=pd.DataFrame({'x':x,'y':y}).dropna()
    return pair.x.corr(pair.y) if len(pair)>3 and pair.x.std()>0 and pair.y.std()>0 else np.nan

def partial_corr(frame,x,y):
    f=frame[[x,y,'market_return','market_volatility']].dropna()
    if len(f)<30:return np.nan
    months=pd.Series([i.month for i in f.index],index=f.index)
    controls=pd.get_dummies(months,prefix='month',drop_first=True,dtype=float)
    controls['market']=f.market_return
    controls['market_volatility']=f.market_volatility
    controls['trend']=np.arange(len(f))/12
    X=np.column_stack([np.ones(len(f)),controls.to_numpy()])
    a=f[x].to_numpy();b=f[y].to_numpy()
    a=a-X@np.linalg.lstsq(X,a,rcond=None)[0]
    b=b-X@np.linalg.lstsq(X,b,rcond=None)[0]
    return corr(a,b)

def year_bootstrap(frame,function,reps=1000):
    rng=np.random.default_rng(1709)
    years=sorted(set(i.year for i in frame.index))
    groups=[frame[np.array([i.year==year for i in frame.index])] for year in years]
    draws=[]
    for _ in range(reps):
        sample=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        result=function(sample)
        if np.isfinite(result):draws.append(result)
    return list(np.quantile(draws,[.025,.975])) if len(draws)>=reps*.8 else [np.nan,np.nan]

def probabilities(panel):
    results=[]
    for asset in INSURERS+['Insurance basket']:
        f=panel[asset].dropna(subset=['price_return','phase'])
        for season in ['All months','Jun–Nov']:
            s=f if season=='All months' else f[f.hurricane_season]
            base=(s.price_return<0).mean()
            for phase in PHASES:
                sub=s[s.phase==phase];n=len(sub);k=int((sub.price_return<0).sum())
                if not n:continue
                p=k/n
                interval=year_bootstrap(s,lambda b: (b.loc[b.phase==phase,'price_return']<0).mean() if (b.phase==phase).any() else np.nan)
                delta=year_bootstrap(s,lambda b: (b.loc[b.phase==phase,'price_return']<0).mean()-(b.price_return<0).mean() if (b.phase==phase).any() else np.nan)
                runs=int((s.phase.eq(phase)&~s.phase.eq(phase).shift(1,fill_value=False)).sum())
                results.append({'asset':asset,'season':season,'phase':phase,'months':n,'negative_months':k,
                   'probability':p,'baseline':base,'difference':p-base,'ci_low':interval[0],'ci_high':interval[1],
                   'difference_low':delta[0],'difference_high':delta[1],'state_runs':runs,
                   'mean_return':sub.price_return.mean(),'mean_volatility':sub.volatility.mean(),
                   'drop_5pct_probability':(sub.price_return<=-.05).mean()})
                results[-1]['total_return_down_probability']=(sub.total_return<0).mean()
    return results

def strategy_test(asset_returns,feature,kind,start='2017-01',lag_label=14):
    common=asset_returns.dropna().index.intersection(feature.index)
    common=common[common>=pd.Period(start)]
    R=asset_returns.loc[common].to_numpy()
    features=feature.loc[common]
    if kind=='insurance':risk=(features.enso<=-.5)&features.hurricane_season
    else:risk=features.enso_abs>=1.0
    exposure=np.where(risk,.5,1.)
    models={}
    # Include a matched-average-exposure control to separate timing from less risk.
    for name,w in [('ENSO overlay',exposure),('Always invested',np.ones(len(common))),('Same average exposure',np.full(len(common),exposure.mean()))]:
        before=np.zeros(R.shape[1]);returns=[];turnovers=[];wealth=1.
        for i in range(len(common)):
            target=np.full(R.shape[1],w[i]/R.shape[1])
            turnover=np.abs(target-before).sum()
            gross=float(target@R[i]);net=(1-.001*turnover)*(1+gross)-1
            returns.append(net);turnovers.append(turnover)
            # Post-return risky weights, before the next monthly rebalance.
            before=target*(1+R[i])/(1+gross)
        r=np.array(returns);path=np.r_[1.,np.cumprod(1+r)]
        models[name]={'cagr':path[-1]**(12/len(r))-1,'annual_volatility':np.std(r,ddof=1)*np.sqrt(12),
            'max_drawdown':float(np.min(path/np.maximum.accumulate(path)-1)),
            'sharpe_zero_cash':np.mean(r)/np.std(r,ddof=1)*np.sqrt(12),
            'total_return':path[-1]-1,'mean_exposure':float(np.mean(w)),'annual_turnover':np.mean(turnovers)*12,
            'months':len(r),'start':str(common[0]),'end':str(common[-1]),'returns':r.tolist(),
            'equity_curve':path[1:].tolist(),'dates':[str(m) for m in common]}
    return {'asset':kind,'lag_days':lag_label,'models':models,'reduced_months':int(risk.sum())}

def calibration(frame):
    f=frame.dropna(subset=['price_return','phase']).sort_index()
    records=[]
    for i in range(len(f)):
        if f.index[i]<pd.Period('2017-01'):continue
        train=f.iloc[:i]
        if len(train)<36:continue
        phase=f.iloc[i].phase
        subset=train[train.phase==phase]
        p0=((train.price_return<0).sum()+1)/(len(train)+2)
        p=((subset.price_return<0).sum()+1)/(len(subset)+2) if len(subset)>=12 else p0
        y=int(f.iloc[i].price_return<0)
        records.append({'month':str(f.index[i]),'p_enso':p,'p_baseline':p0,'negative':y})
    a=pd.DataFrame(records)
    bs=((a.p_enso-a.negative)**2).mean();base=((a.p_baseline-a.negative)**2).mean()
    return {'brier_enso':bs,'brier_baseline':base,'skill':1-bs/base,'months':len(a),'predictions':records}

def main():
    stocks={s:stock(s) for s in INSURERS+['SPY']+FUNDS}
    fut={};quality={}
    for code in FUTURES:fut[code],quality[code]=futures(code)
    weekly,oni=parse_climate()
    assets={**stocks,**fut}
    monthly_data={s:monthly(frame) for s,frame in assets.items()}
    # Basket monthly returns are equal-weight, monthly rebalanced price returns.
    for label,members in [('Insurance basket',INSURERS),('Agriculture quote basket',list(FUTURES))]:
        returns=pd.concat({s:assets[s].price_return for s in members},axis=1,sort=True).dropna()
        daily_basket=returns.mean(axis=1)
        basket=pd.DataFrame({'close':(1+daily_basket).cumprod(),'price_return':daily_basket,'total_return':np.nan})
        if label=='Insurance basket':
            basket['total_return']=pd.concat({s:assets[s].total_return for s in members},axis=1).dropna().mean(axis=1)
        m=monthly(basket)
        m['price_return']=pd.concat({s:monthly_data[s].price_return for s in members},axis=1).dropna().mean(axis=1)
        if label=='Insurance basket':
            m['total_return']=pd.concat({s:monthly_data[s].total_return for s in members},axis=1).dropna().mean(axis=1)
        monthly_data[label]=m
    months=pd.period_range('2006-10','2026-08',freq='M')
    features=climate_features(months,stocks['SPY'].index,weekly,oni)
    market=monthly_data['SPY'][['price_return','volatility']].rename(columns={'price_return':'market_return','volatility':'market_volatility'})
    panel={s:m.join(features,how='inner').join(market).dropna(subset=['price_return','enso']) for s,m in monthly_data.items()}
    correlation=[]
    for asset,f in panel.items():
        if asset=='SPY':continue
        r1=corr(f.enso,f.price_return);rv=corr(f.enso,f.volatility);ra=corr(f.enso_abs,f.volatility)
        correlation.append({'asset':asset,'months':len(f),'enso_return':r1,'enso_volatility':rv,'strength_volatility':ra,
            'spearman_strength_volatility':spearmanr(f.enso_abs,f.volatility,nan_policy='omit').statistic,
            'partial_strength_volatility':partial_corr(f,'enso_abs','volatility'),
            'partial_enso_return':partial_corr(f,'enso','price_return'),
            'oni_return':corr(f.oni,f.price_return),'oni_strength_volatility':corr(f.oni.abs(),f.volatility)})
    corr_intervals=[]
    for asset in ['Insurance basket','Agriculture quote basket','DBA']:
        f=panel[asset]
        for x,y in [('enso','price_return'),('enso','volatility'),('enso_abs','volatility')]:
            ci=year_bootstrap(f,lambda b:corr(b[x],b[y]))
            corr_intervals.append({'asset':asset,'x':x,'y':y,'r':corr(f[x],f[y]),'ci_low':ci[0],'ci_high':ci[1]})
    cross=[]
    for insurance in INSURERS+['Insurance basket']:
        for ag in list(FUTURES)+['Agriculture quote basket','DBA']:
            f=pd.concat([panel[insurance].volatility.rename('insurance_vol'),panel[ag].volatility.rename('ag_vol'),
                         features.enso,features.enso_abs,market],axis=1).dropna()
            cross.append({'insurance':insurance,'agriculture':ag,'months':len(f),'volatility_correlation':corr(f.insurance_vol,f.ag_vol),
                          'partial_market_season_trend':partial_corr(f,'insurance_vol','ag_vol')})
    probability=probabilities(panel)
    insurance_returns=pd.concat({s:monthly_data[s].total_return for s in INSURERS},axis=1)
    tests=[]
    for lag in [14,28]:
        feat=climate_features(months,stocks['SPY'].index,weekly,oni,lag)
        tests.append(strategy_test(insurance_returns,feat,'insurance',lag_label=lag))
        for fund in FUNDS:
            test=strategy_test(monthly_data[fund][['total_return']],feat,'agriculture',lag_label=lag)
            test['asset']=fund;tests.append(test)
    probability_calibration={s:calibration(panel[s]) for s in INSURERS+['Insurance basket']}
    coverage=[]
    for symbol,frame in assets.items():
        coverage.append({'asset':symbol,'name':FUTURES.get(symbol,symbol),'source':'Yahoo Finance public futures' if symbol in FUTURES else 'Massive',
            'start':str(frame.index.min().date()),'end':str(frame.index.max().date()),'daily_rows':len(frame),
            'monthly_rows':len(panel.get(symbol,[])),'data_kind':'continuous/nearby quote history; roll unverified' if symbol in FUTURES else 'split-adjusted equity/fund prices plus cash distributions'})
    for symbol,frame in assets.items():
        export=frame.copy();export.index.name='date';export.reset_index().to_json(CALC/('daily_'+symbol+'.jsonl'),orient='records',lines=True,date_format='iso')
    for symbol,frame in panel.items():
        export=frame.copy();export.index=export.index.astype(str);export.reset_index().to_json(CALC/('monthly_'+symbol.replace(' ','_')+'.jsonl'),orient='records',lines=True,date_format='iso')
    weekly.to_json(CALC/'weekly_enso.jsonl',orient='records',lines=True,date_format='iso')
    oni.to_json(CALC/'oni.jsonl',orient='records',lines=True,date_format='iso')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file()}
    summary={'created_at':datetime.now(timezone.utc).isoformat(),'coverage':coverage,'futures_quality':quality,
         'correlations':correlation,'correlation_intervals':corr_intervals,'volatility_cross_correlations':cross,
         'downside_probabilities':probability,'backtests':tests,'probability_calibration':probability_calibration,
         'weekly_enso_rows':len(weekly),'weekly_enso_first':str(weekly.center_date.min().date()),
         'weekly_enso_last':str(weekly.center_date.max().date()),'oni_rows':len(oni),'raw_sha256':hashes}
    save('results.json',summary)
    print(json.dumps(clean_json({'coverage':coverage,'correlations':correlation,'main_intervals':corr_intervals,
          'backtests':[{**{k:v for k,v in t.items() if k!='models'},'models':{n:{k:v for k,v in m.items() if k not in ['returns','equity_curve','dates']} for n,m in t['models'].items()}} for t in tests],
          'probability_basket':[p for p in probability if p['asset']=='Insurance basket'],
          'calibration':{k:{a:b for a,b in v.items() if a!='predictions'} for k,v in probability_calibration.items()}}),indent=2))

if __name__=='__main__':main()
