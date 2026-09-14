"""Build a self-contained HTML report with embedded Plotly; no live requests."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parent
C=ROOT/'calculated'
R=json.loads((C/'results.json').read_text())
COLORS={'Cold':'#3477bc','Neutral':'#7b8895','Warm':'#d26b45','ENSO overlay':'#087e8b','Always invested':'#172e4d','Same average exposure':'#b48b48'}

def pct(x,d=1):return '—' if x is None else f'{100*x:.{d}f}%'
def num(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def pp(x):return '—' if x is None else f'{100*x:+.1f} pp'
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def chart(fig,title,height=430):
    fig.update_layout(template='plotly_white',height=height,title={'text':title,'font':{'size':18,'color':'#172e4d'}},
       font={'family':'Arial, sans-serif','size':12,'color':'#33465a'},paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',
       margin={'l':60,'r':25,'t':65,'b':65},legend={'orientation':'h','y':1.10},hovermode='closest')
    return fig.to_html(full_html=False,include_plotlyjs=False,config={'responsive':True,'displaylogo':False,'scrollZoom':False})

def month_data(name):
    d=pd.read_json(C/('monthly_'+name.replace(' ','_')+'.jsonl'),lines=True)
    d['date']=pd.to_datetime(d.month+'-01');return d

def make_charts():
    charts={}
    weekly=pd.read_json(C/'weekly_enso.jsonl',lines=True)
    weekly['center_date']=pd.to_datetime(weekly.center_date)
    weekly=weekly[weekly.center_date>='2006-09-10']
    f=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.10,subplot_titles=['Weekly Niño 3.4 anomaly (°C)','Insurance: next-month realised volatility','Agricultural quotes: next-month realised volatility'])
    f.add_trace(go.Scatter(x=weekly.center_date,y=weekly.nino34,name='Weekly Niño 3.4',line={'color':'#172e4d','width':1.5}),row=1,col=1)
    for level in [-.5,.5]:f.add_hline(y=level,line_dash='dot',line_color='#a7b0ba',row=1,col=1)
    for row,name,color in [(2,'Insurance basket','#087e8b'),(3,'Agriculture quote basket','#d26b45')]:
        d=month_data(name)
        f.add_trace(go.Scatter(x=d.date,y=d.volatility*100,name=name,line={'color':color,'width':1.6}),row=row,col=1)
        f.update_yaxes(ticksuffix='%',row=row,col=1)
    f.update_layout(showlegend=False)
    charts['timeline']=chart(f,'20 years of climate and market variability',690)

    assets=['RNR','EG','ACGL','Insurance basket']+[x['asset'] for x in R['correlations'] if x['asset'] not in ['RNR','EG','ACGL','Insurance basket','Agriculture quote basket','DBA','CORN','SOYB','WEAT']]+['Agriculture quote basket','DBA']
    lookup={x['asset']:x for x in R['correlations']}
    fields=['enso_return','enso_volatility','strength_volatility','partial_strength_volatility']
    z=[[lookup[a][v] for v in fields] for a in assets]
    fig=go.Figure(go.Heatmap(z=z,y=assets,x=['ENSO → return','ENSO → volatility','|ENSO| → volatility','|ENSO| → volatility<br>after controls'],zmin=-.5,zmax=.5,
         colorscale=[[0,'#3477bc'],[.5,'#f7f7f2'],[1,'#d26b45']],text=[[f'{v:.2f}' for v in row] for row in z],texttemplate='%{text}',colorbar={'title':'r'},hovertemplate='%{y}<br>%{x}: %{z:.3f}<extra></extra>'))
    fig.update_yaxes(autorange='reversed')
    charts['heatmap']=chart(fig,'Lagged ENSO correlations with the following month',790)

    fig=go.Figure()
    for phase in ['Cold','Neutral','Warm']:
        records=[x for x in R['downside_probabilities'] if x['season']=='All months' and x['phase']==phase]
        fig.add_trace(go.Bar(name=phase,x=[x['asset'] for x in records],y=[100*x['probability'] for x in records],marker_color=COLORS[phase],
          error_y={'type':'data','symmetric':False,'array':[100*(x['ci_high']-x['probability']) for x in records],'arrayminus':[100*(x['probability']-x['ci_low']) for x in records]},
          customdata=[[x['negative_months'],x['months']] for x in records],hovertemplate='%{x}: %{y:.1f}%<br>%{customdata[0]} negative / %{customdata[1]} months<extra>%{fullData.name}</extra>'))
    fig.update_layout(barmode='group');fig.update_yaxes(title='Probability of next-month price decline',ticksuffix='%',range=[0,70])
    charts['probabilities']=chart(fig,'Downside frequencies overlap substantially',450)

    names=['ZC','ZS','ZW','ZM','ZL','ZO','ZR','KE','LE','GF','HE','CC','KC','CT','SB','OJ','Agriculture quote basket','DBA']
    ins=['RNR','EG','ACGL','Insurance basket'];lookup={(x['insurance'],x['agriculture']):x for x in R['volatility_cross_correlations']}
    fig=go.Figure(go.Heatmap(z=[[lookup[(i,a)]['volatility_correlation'] for a in names] for i in ins],x=names,y=ins,zmin=-.5,zmax=.8,
         colorscale=[[0,'#3477bc'],[.385,'#f7f7f2'],[1,'#d26b45']],colorbar={'title':'r'},hovertemplate='%{y} / %{x}<br>Volatility correlation: %{z:.3f}<extra></extra>'))
    charts['cross']=chart(fig,'Insurance and agricultural volatility move together—but that is not proof of ENSO causation',360)

    for key,title in [('insurance','Insurance basket'),('DBA','DBA agriculture futures fund')]:
        test=next(x for x in R['backtests'] if x['asset']==key and x['lag_days']==14)
        fig=go.Figure()
        for name,m in test['models'].items():
            fig.add_trace(go.Scatter(x=m['dates'],y=np.array(m['equity_curve'])*100,name=name,line={'color':COLORS[name],'width':2,'dash':'dot' if name=='Same average exposure' else 'solid'}))
        fig.update_yaxes(title='Growth of 100, after modelled costs')
        charts[key]=chart(fig,title+': 2017–2026 later-period backtest',430)
    return charts

def main():
    charts=make_charts()
    robustness=[]
    for asset in ['Insurance basket','Agriculture quote basket','DBA']:
        data=month_data(asset)
        crisis=((data.month>='2008-01')&(data.month<='2009-06'))|((data.month>='2020-02')&(data.month<='2020-12'))
        for label,subset in [('2006–2016',data[data.month<'2017-01']),('2017–2026',data[data.month>='2017-01']),('Without specified crisis windows',data[~crisis])]:
            robustness.append({'asset':asset,'sample':label,'months':len(subset),'enso_return':subset.enso.corr(subset.price_return),
                 'enso_volatility':subset.enso.corr(subset.volatility),'strength_volatility':subset.enso_abs.corr(subset.volatility)})
    (C/'robustness.json').write_text(json.dumps(robustness,indent=2))
    robustness_table=table(['Series','Subsample','Months','ENSO / return','ENSO / volatility','|ENSO| / volatility'],[
        [x['asset'],x['sample'],x['months'],num(x['enso_return']),num(x['enso_volatility']),num(x['strength_volatility'])] for x in robustness])
    corr={x['asset']:x for x in R['correlations']}
    cross=next(x for x in R['volatility_cross_correlations'] if x['insurance']=='Insurance basket' and x['agriculture']=='Agriculture quote basket')
    tests={x['asset']:x for x in R['backtests'] if x['lag_days']==14}
    insurance=tests['insurance']['models'];dba=tests['DBA']['models']
    baseline=next(x['baseline'] for x in R['downside_probabilities'] if x['asset']=='Insurance basket' and x['season']=='All months')
    prob_rows=[]
    for p in R['downside_probabilities']:
        if p['season']=='All months':
            prob_rows.append([p['asset'],p['phase'],f"{p['negative_months']} / {p['months']}",pct(p['probability']),f"{pct(p['ci_low'])}–{pct(p['ci_high'])}",pct(p['baseline']),pp(p['difference']),pct(p['drop_5pct_probability'])])
    prob_table=table(['Asset','ENSO proxy','Declines / months','P(price decline)','95% year-block interval','Unconditional','Difference','P(drop ≥5%)'],prob_rows)
    hurricane_table=table(['ENSO proxy','Declines / months','P(basket price decline)','95% interval','Mean realised volatility'],[
        [p['phase'],f"{p['negative_months']} / {p['months']}",pct(p['probability']),f"{pct(p['ci_low'])}–{pct(p['ci_high'])}",pct(p['mean_volatility'])]
        for p in R['downside_probabilities'] if p['asset']=='Insurance basket' and p['season']=='Jun–Nov'])
    interval_table=table(['Series','Relationship','Pearson r','95% year-block interval'],[
        [x['asset'],{'enso':'Signed ENSO','enso_abs':'Absolute ENSO'}[x['x']]+' → '+{'price_return':'next-month price return','volatility':'next-month volatility'}[x['y']],num(x['r']),f"{num(x['ci_low'])} to {num(x['ci_high'])}"] for x in R['correlation_intervals']])
    all_corr_table=table(['Series','Months','ENSO / return','ENSO / vol','|ENSO| / vol','Spearman |ENSO| / vol','Controlled |ENSO| / vol','ONI / return','|ONI| / vol'],[
        [x['asset'],x['months']]+[num(x[k]) for k in ['enso_return','enso_volatility','strength_volatility','spearman_strength_volatility','partial_strength_volatility','oni_return','oni_strength_volatility']] for x in R['correlations']])
    comparison=[]
    for asset,t in tests.items():
        for name,m in t['models'].items():
            comparison.append([asset.title() if asset=='insurance' else asset,name,pct(m['cagr']),pct(m['annual_volatility']),pct(m['max_drawdown']),num(m['sharpe_zero_cash'],2),pct(m['mean_exposure'])])
    strategy_table=table(['Portfolio','Rule','Annualised return','Annual volatility','Max drawdown','Sharpe (cash=0)','Avg risky exposure'],comparison)
    sensitivity_table=table(['Portfolio','14-day overlay CAGR','28-day overlay CAGR','Always invested CAGR'],[
        [a,pct(tests[a]['models']['ENSO overlay']['cagr']),pct(next(x for x in R['backtests'] if x['asset']==a and x['lag_days']==28)['models']['ENSO overlay']['cagr']),pct(tests[a]['models']['Always invested']['cagr'])] for a in tests])
    calibration_table=table(['Asset','ENSO Brier score','Baseline Brier score','Skill vs baseline','Later-period months'],[
        [a,num(v['brier_enso'],4),num(v['brier_baseline'],4),pct(v['skill']),v['months']] for a,v in R['probability_calibration'].items()])
    coverage_table=table(['Symbol','Instrument','Source','First daily record','Last daily record','Daily rows','Usable months'],[
        [html.escape(x['asset']),html.escape(x['name']),x['source'],x['start'],x['end'],f"{x['daily_rows']:,}",x['monthly_rows']] for x in R['coverage']])
    quality_table=table(['Futures code','Usable daily rows','Weekend rows removed','Duplicate dates removed','Zero-volume rows retained','Daily quote moves >10%'],[
        [a,v['usable_rows'],v['weekend_rows_removed'],v['duplicate_dates_removed'],v['zero_volume_rows'],v['moves_over_10pct']] for a,v in R['futures_quality'].items()])
    total_rows=sum(x['daily_rows'] for x in R['coverage'])
    html_body=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ENSO, insurance & agriculture — historical research</title>
<style>
:root{{--ink:#172e4d;--muted:#58697a;--teal:#087e8b;--warm:#d26b45;--paper:#f4f5f1;--line:#dde3e5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}a{{color:#087583;text-underline-offset:3px}}header{{background:var(--ink);color:white;padding:64px max(24px,calc((100vw - 1180px)/2)) 52px}}header .eyebrow{{color:#89d5d5;text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:700}}h1{{font-size:clamp(35px,5vw,62px);line-height:1.08;letter-spacing:-.035em;max-width:900px;margin:20px 0}}header p{{max-width:880px;color:#cfdae4;font-size:19px}}header .meta{{font-size:13px;margin-top:26px;color:#aebfd0}}main{{max-width:1228px;margin:auto;padding:28px 24px 80px}}nav{{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:0 0 28px}}section{{background:white;border:1px solid var(--line);border-radius:14px;padding:34px;margin:22px 0}}h2{{font-size:29px;line-height:1.2;letter-spacing:-.025em;margin:0 0 20px}}h3{{font-size:19px;margin:26px 0 10px}}p{{margin:12px 0}}.lead{{font-size:21px;line-height:1.5}}.muted,.caption{{color:var(--muted);font-size:14px}}.tag{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:750;margin-bottom:12px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.metric{{padding:22px;background:#f3f7f7;border-radius:10px}}.metric strong{{display:block;font-size:35px;line-height:1.3}}.metric span{{display:block;font-size:13px;color:var(--muted)}}.callout{{border-left:4px solid var(--warm);background:#fcf4ee;padding:18px 22px;margin:22px 0}}.positive{{border-left-color:var(--teal);background:#eff8f7}}.table-wrap{{overflow:auto;margin:18px 0;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}}th{{text-align:left;padding:12px;background:#eef3f5;white-space:nowrap}}td{{padding:11px 12px;border-top:1px solid #e7ebed;vertical-align:top}}tbody tr:nth-child(even){{background:#fafbf9}}td:not(:first-child){{font-variant-numeric:tabular-nums}}.formula{{background:#f1f5f6;padding:15px 18px;border-radius:8px;margin:13px 0;overflow:auto;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px;line-height:1.8}}.formula b{{color:#087e8b}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}select{{padding:10px 12px;background:white;border:1px solid #9aabb6;border-radius:6px;font:inherit;margin:0 12px 16px 0}}label{{display:inline-block;font-size:13px;color:var(--muted)}}.scenario-cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.scenario{{background:#f3f7f7;border-radius:8px;padding:18px}}.scenario strong{{display:block;font-size:29px}}.scenario small{{display:block;color:var(--muted);font-size:12px}}ul,ol{{padding-left:23px}}li{{margin:8px 0}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:700;color:var(--teal)}}code{{font-size:13px}}.sources li{{font-size:14px}}footer{{font-size:12px;color:var(--muted);padding:20px 0}}.plotly-graph-div{{max-width:100%}}@media(max-width:760px){{section{{padding:22px 14px}}main{{padding:20px 12px}}header{{padding:38px 24px}}.grid,.twocol{{grid-template-columns:1fr}}.scenario-cards{{grid-template-columns:repeat(2,1fr)}}h2{{font-size:24px}}}}@media print{{body{{background:white}}header{{padding:25px;color:#172e4d;background:white}}header p,header .meta{{color:#58697a}}main{{max-width:none;padding:0}}section{{break-inside:auto;border:0;padding:20px 0}}nav,.modebar{{display:none}}details{{display:block}}a{{color:#172e4d}}}}
</style><script>{get_plotlyjs()}</script></head><body>
<header><div class="eyebrow">Historical market research · reproducible data study</div><h1>ENSO, insurance<br>& agricultural markets</h1><p>A simple test of climate-state correlations, downside probabilities and defensive exposure rules. The climate mechanism is plausible; the standalone timing signal is weak.</p><div class="meta">Requested: 10 Sep 2006–9 Sep 2026 · Full-month analysis: Oct 2006–Aug 2026 · Later-period tests: Jan 2017–Aug 2026 · Prepared 11 Sep 2026</div></header>
<main><nav><a href="#findings">Findings</a><a href="#probability">Downside probabilities</a><a href="#correlation">Correlations</a><a href="#strategy">Strategy & backtest</a><a href="#methods">Formulas</a><a href="#data">Data & sources</a></nav>
<section id="findings"><div class="tag">What the data says</div><h2>Do not use ENSO alone to predict insurer sell-offs.</h2><p class="lead">Across 239 complete months, the equal-weight RNR / EG / ACGL basket has an ENSO–next-month price-return correlation of <b>{num(corr['Insurance basket']['enso_return'])}</b>. The simple climate-state probability model also performs worse than its unconditional baseline in the later-period test.</p>
<div class="grid"><div class="metric"><strong>{pct(baseline)}</strong><span>Insurance basket: historical probability of a negative month, all ENSO states</span></div><div class="metric"><strong>{num(corr['Insurance basket']['enso_volatility'])}</strong><span>Signed ENSO vs next-month insurance realised volatility</span></div><div class="metric"><strong>{num(cross['volatility_correlation'])}</strong><span>Insurance vs agriculture basket volatility, same month</span></div></div>
<p>Signed ENSO is modestly negatively associated with subsequent volatility: warmer Niño 3.4 readings tend to precede calmer months in these samples. That does <em>not</em> mean strong ENSO events in either direction reliably raise volatility: correlations using absolute ENSO strength are small and unstable. Nor does volatility identify whether prices will rise or fall.</p>
<p>In January 2017–August 2026, the insurer overlay earned <b>{pct(insurance['ENSO overlay']['cagr'],2)}</b> annualised versus <b>{pct(insurance['Always invested']['cagr'],2)}</b> always invested. The DBA overlay earned <b>{pct(dba['ENSO overlay']['cagr'],2)}</b> versus <b>{pct(dba['Always invested']['cagr'],2)}</b>. Both also trailed controls with the same average risky allocation.</p>
<div class="callout"><b>Two material limits.</b> Your Massive key returned HTTP 403 for futures metadata, contracts and bars. Agricultural quote correlations therefore use public Yahoo Finance futures histories; executable strategy simulations use separately labelled futures-based funds. NOAA files are the currently published historical series, not archived release vintages. Delaying them reduces timing leakage but does not remove later revisions or retrospective climatology changes. These are exploratory historical tests, not a verified point-in-time trading record.</div>
{charts['timeline']}<p class="caption">Weekly climate readings are shown at their observation centres; market volatility is plotted in its outcome month. Statistical joins use the lag described below. Equity volatility includes cash distributions; futures quote volatility is subject to unverified contract rolls.</p></section>

<section id="probability"><div class="tag">Conditional risk, not causation</div><h2>How often did insurer prices decline under each ENSO scenario?</h2><p>The outcome is a <b>negative next-calendar-month split-adjusted price return</b>. Cold means the latest eligible weekly Niño 3.4 anomaly is ≤ −0.5°C; warm means ≥ +0.5°C; neutral is between. These are La Niña-like and El Niño-like <em>proxies</em>, not NOAA's official multi-season event declarations.</p>
<label>Climate state<br><select id="phase"><option>Cold</option><option>Neutral</option><option>Warm</option></select></label><label>Outcome months<br><select id="season"><option>All months</option><option>Jun–Nov</option></select></label><div class="scenario-cards" id="scenario-cards"></div>
<p class="caption">Intervals resample complete calendar years 1,000 times. Selecting a scenario displays observed historical frequencies; it does not create a causal forecast or condition on today's valuations, exposures or storm forecasts.</p>
{charts['probabilities']}{prob_table}
<p>The basket's cold, neutral and warm-state estimates are roughly 33%, 35% and 37%. Their uncertainty intervals overlap. All three basket differences from the unconditional baseline also have year-bootstrap intervals containing zero. The data does not support the claim that a cold ENSO state, by itself, increases the chance of an insurer stock-price decline.</p>
<h3>Atlantic hurricane-season subset: June–November</h3>{hurricane_table}
<p>NOAA describes a physical mechanism through which El Niño tends to suppress Atlantic hurricanes via stronger vertical wind shear. Insurer equity prices are further removed: geographically diversified underwriting, reinsurance protection, premiums, investment income, interest rates and expectations all matter. The relevant quantity here is an association in stock outcomes, not the probability of hurricane damage. <a href="https://www.aoml.noaa.gov/how-does-el-nino-impact-atlantic-hurricane-season/">NOAA mechanism</a>.</p>
<div class="formula">Estimated: <b>P(R<sub>next month</sub> &lt; 0 | ENSO proxy = s)</b><br>Not identified: P(the decline was <b>caused by</b> ENSO)</div>
<p>Price-decline counts include ex-dividend price adjustments; strategy returns below add cash distributions. A special dividend can reduce a quoted price without creating the same economic loss. The calculated results also retain total-return downside frequencies for that sensitivity.</p></section>

<section id="correlation"><div class="tag">Simple correlations</div><h2>ENSO direction, ENSO strength and volatility are different questions.</h2><p>Each row pairs a lagged climate measurement with the following month's price return or annualised daily realised volatility. We avoid correlations between raw trending price <em>levels</em> and climate indices. Agricultural rows cover 16 benchmark series, not every agricultural contract worldwide.</p>
{charts['heatmap']}<p class="caption">Pearson r ranges from −1 to +1. Colours are centred at zero and clipped at ±0.5 for readability. “After controls” removes broad equity-market return and volatility, calendar-month effects and a linear trend from both variables. Those contemporaneous controls are explanatory diagnostics, not inputs to the trading rule.</p>
{interval_table}<p>These are pointwise, exploratory intervals. The report screens many related relationships; intervals are not adjusted for multiple comparisons and should not be read as discovery of statistically proven trading signals. Whole-year resampling preserves within-year clustering, but may still understate uncertainty across multi-year ENSO events.</p>
<h3>Does the relationship persist in the later period?</h3>{robustness_table}<p>The signed ENSO–volatility relationships weaken substantially in 2017–2026: about −0.06 for insurers and −0.07 for the agriculture quote basket; DBA changes sign. This instability is another reason not to treat the full-period correlation as a reliable forecasting rule. The final diagnostic excludes January 2008–June 2009 and February–December 2020 to check sensitivity to large financial crises; it is an explanatory subsample, not a trading filter or a tuned strategy.</p>
<details><summary>Full results, including rank correlations and the slower ONI measure</summary>{all_corr_table}<p class="caption">ONI is read independently from the NOAA seasonal file and delayed until 15 days after the end of its full three-month observation window. It is not the same series as a weekly Niño 3.4 anomaly. Differences between ONI and weekly results are a useful stability warning.</p></details>
<h3>Insurance / agriculture volatility correlation</h3>{charts['cross']}
<p>The basket-to-basket contemporaneous correlation is <b>{num(cross['volatility_correlation'])}</b>; after market return, market volatility, calendar-month and trend controls it is <b>{num(cross['partial_market_season_trend'])}</b>. Shared financial stress can make both markets volatile. This relationship is not evidence that ENSO is the common cause, and it does not establish a directional hedge.</p></section>

<section id="strategy"><div class="tag">Specified rules · no parameter search</div><h2>A defensive ENSO overlay fails the performance test here.</h2><div class="twocol"><div><h3>Insurance rule</h3><p>At the final trading close of each month, hold an equal-weight basket of RNR, EG and ACGL. For the <em>upcoming</em> June–November month, reduce total risky exposure to 50% if the eligible weekly Niño 3.4 anomaly is ≤ −0.5°C. Otherwise hold 100%. The unused weight stays in cash.</p></div><div><h3>Agriculture rule</h3><p>At the same monthly schedule, hold 50% rather than 100% when |Niño 3.4| ≥ 1.0°C, testing the simple idea that stronger climate anomalies warrant less exposure. Evaluate independently on DBA, CORN, SOYB and WEAT. No short selling or futures leverage is simulated.</p></div></div>
<p>Both rules were specified before calculating results; thresholds were not selected to maximise this sample. Models use a 14-calendar-day delay from each weekly observation centre, with 28 days as a sensitivity. Evaluation is January 2017–August 2026; the earlier period supports probability estimates. This temporal separation does not make the current-vintage climate series historically unrevised.</p>
<p>Returns include available cash distributions reinvested on ex-date and are net of 10 basis points per dollar of risky-asset turnover, including initial entry and monthly rebalancing. Fractional shares, no taxes and zero cash yield are assumed. Fund expenses and fund-level roll effects are embedded in observed share returns. The final position is marked to market, not liquidated. The 10 bp charge is a modelling assumption, not an observed execution quote.</p>
{charts['insurance']}{charts['DBA']}{strategy_table}
<p>The equal-average-exposure comparator separates the effect of holding less risk from the value of timing it. Its fixed weight is the overlay's realised average weight across the evaluation period, so it is an <em>ex-post diagnostic</em>, not a preselected deployable allocation.</p>
<h3>Delay sensitivity</h3>{sensitivity_table}
<h3>Does ENSO improve downside probability forecasts?</h3><p>For each evaluation month, estimate the negative-month rate using only earlier months in the same climate state, with simple add-one smoothing. Compare it with the similarly smoothed unconditional rate from the same past data. Lower Brier score is better; negative skill means ENSO made the forecast worse.</p>{calibration_table}
<div class="callout positive"><b>Research decision:</b> retain ENSO as a contextual risk indicator, not a standalone buy/sell trigger. Neither the insurer rule nor the agriculture rule demonstrates a return advantage in this evaluation. A reduction in volatility alone is expected when exposure is reduced and is insufficient evidence of useful timing.</div>
<p>We do not label a stitched nearby-futures quote path as an investable return. Without contract identities, rolls and transaction records, a change across delivery months may be a curve-price discontinuity. Direct futures execution backtests require your futures entitlement or a verified contract-level archive. <a href="https://teucrium.com/corn">Teucrium CORN</a> and <a href="https://www.invesco.com/content/dam/invesco/us/en/product-documents/etf/fact-sheet/dba-invesco-db-agriculture-fund-fact-sheet.pdf">Invesco DBA</a> explain their futures exposure.</p></section>

<section id="methods"><div class="tag">Formulas and reproducibility</div><h2>Every calculation, in plain terms.</h2>
<h3>1. Price changes, distributions and volatility</h3><div class="formula">r<sub>d</sub><sup>price</sup> = P<sub>d</sub><sup>adj</sup> / P<sub>d−1</sub><sup>adj</sup> − 1<br>D<sub>d</sub><sup>adj</sup> = D<sub>d</sub> × ∏<sub>splits after d</sub>(split_from / split_to)<br>r<sub>d</sub><sup>total</sup> = (P<sub>d</sub><sup>adj</sup> + D<sub>d</sub><sup>adj</sup>) / P<sub>d−1</sub><sup>adj</sup> − 1<br>R<sub>m</sub><sup>price</sup> = P<sub>last,m</sub><sup>adj</sup> / P<sub>last,m−1</sub><sup>adj</sup> − 1<br>R<sub>m</sub><sup>total</sup> = ∏<sub>d in m</sub>(1 + r<sub>d</sub><sup>total</sup>) − 1<br>ℓ<sub>d</sub> = ln(1 + r<sub>d</sub>)<br>σ<sub>m</sub> = √252 × √[Σ<sub>d in m</sub>(ℓ<sub>d</sub> − mean(ℓ))² / (n<sub>m</sub> − 1)]</div>
<p>P<sup>adj</sup> is split-adjusted, not automatically dividend-adjusted. D is cash per share. Equity/fund volatility uses total daily returns; futures quote volatility uses daily quoted-price returns. 252 is an annualisation convention. Basket price returns are the equal-weight average of constituents' monthly price returns. Basket realised volatility uses equal-weight daily constituent returns, so it describes a daily-rebalanced risk basket and is not exactly the volatility of the monthly-rebalanced strategy.</p>
<h3>2. ENSO measures and timing</h3><div class="formula">Niño 3.4 anomaly = regional SST − reference-climatology SST<br>ONI<sub>season</sub> = mean(three monthly Niño 3.4 SST anomalies), as published by NOAA<br>X<sub>m</sub> = latest weekly anomaly with centre_date + 14 days ≤ decision_date<sub>m</sub><br>Cold: X ≤ −0.5°C; Neutral: −0.5°C &lt; X &lt; +0.5°C; Warm: X ≥ +0.5°C<br>Strength = |X|</div><p>The decision date precedes the outcome month and is the previous month's last SPY trading date. The 14-day and 28-day delays are conservative research assumptions, not verified historical release timestamps. Stale weekly measurements more than 45 days old are rejected. A centred weekly reading includes days after its centre; it is never used on the centre date. A DJF ONI reading includes February and is not treated as a January signal. ONI is eligible only after the full season ends plus a 15-day assumed delay. No daily ENSO values are invented by interpolation.</p>
<h3>3. Correlation and controls</h3><div class="formula">ρ(X,Y) = Σ(X−X̄)(Y−Ȳ) / √[Σ(X−X̄)² × Σ(Y−Ȳ)²]<br>Spearman ρ = Pearson correlation of the ranks of X and Y<br>Z = [intercept, SPY monthly return, SPY realised volatility, 11 month dummies, linear trend]<br>e<sub>X</sub> = X − Z(Z⁺X); e<sub>Y</sub> = Y − Z(Z⁺Y)<br>Partial correlation = ρ(e<sub>X</sub>, e<sub>Y</sub>)</div><p>Z⁺ denotes the least-squares pseudoinverse. Controls diagnose seasonality/common-market effects but cannot remove every confounder. ENSO correlations pair X from the decision date with the subsequent month's outcome. Insurance/agriculture volatility correlations pair the two volatilities in the same month. Rank correlation checks sensitivity to scale and extremes.</p>
<h3>4. Scenario frequencies and uncertainty</h3><div class="formula">p̂<sub>s</sub> = number of months with (state=s and R&lt;0) / number of months with state=s<br>Δp<sub>s</sub> = p̂<sub>s</sub> − p̂<sub>all</sub><br>95% interval = [2.5th percentile, 97.5th percentile] of 1,000 year-block resamples</div><p>A bootstrap sample draws whole observed calendar years with replacement, retaining each selected year's months. The same sample is used for a conditional-minus-baseline difference. The seed is fixed at 1709. About 20 years supply far fewer independent ENSO events than the monthly row count suggests. Results are associations conditional on the chosen, surviving insurers; they do not estimate causation or a complete historical insurance industry.</p>
<h3>5. Portfolio rules, costs and performance</h3><div class="formula">w<sub>insurance,m</sub> = 0.5 if (X<sub>m</sub>≤−0.5 and outcome_month∈Jun…Nov), else 1<br>w<sub>agriculture,m</sub> = 0.5 if |X<sub>m</sub>|≥1.0, else 1<br>target risky weight a<sub>i,m</sub> = w<sub>m</sub>/N<br>turnover<sub>m</sub> = Σ<sub>i</sub>|a<sub>i,m</sub> − pretrade_weight<sub>i,m</sub>|<br>g<sub>m</sub> = Σ<sub>i</sub>a<sub>i,m</sub>R<sub>i,m</sub><sup>total</sup><br>R<sub>strategy,m</sub> = (1 − 0.001×turnover<sub>m</sub>)(1 + g<sub>m</sub>) − 1<br>pretrade_weight<sub>i,m+1</sub> = a<sub>i,m</sub>(1+R<sub>i,m</sub><sup>total</sup>)/(1+g<sub>m</sub>)<br>V<sub>m</sub> = V<sub>m−1</sub>(1+R<sub>strategy,m</sub>), V<sub>0</sub>=100<br>CAGR = (V<sub>T</sub>/V<sub>0</sub>)<sup>12/T</sup> − 1<br>Annual strategy volatility = √12 × sample_std(monthly strategy returns)<br>Max drawdown = min<sub>m</sub>[V<sub>m</sub>/max<sub>j≤m</sub>V<sub>j</sub> − 1]<br>Sharpe (zero cash rate) = √12 × mean(monthly returns)/sample_std(monthly returns)</div><p>T counts evaluation months. The same turnover model applies to all comparators. Initial pretrade risky weights are zero. Prices are historical closing prices, with the specified cost allowance rather than executable bid/ask fills. No leverage, financing charges, borrowing, tax or liquidation cost is modelled.</p>
<h3>6. Probability backtest</h3><div class="formula">p<sub>m,s</sub> = (previous negative months in state s + 1)/(previous months in state s + 2)<br>Brier score = mean[(p<sub>m</sub> − 1{{R<sub>m</sub>&lt;0}})²]<br>Brier skill = 1 − Brier<sub>ENSO</sub>/Brier<sub>unconditional</sub></div><p>At least 36 past months are required overall; with fewer than 12 past observations in a state, the state forecast falls back to the unconditional rate. No evaluation-month outcome is used to estimate its own probability. Both probability models are updated sequentially.</p></section>

<section id="data"><div class="tag">Coverage and audit</div><h2>What was actually downloaded and used.</h2><p><b>{total_rows:,} usable daily price rows</b> cover three insurers, SPY, four agricultural futures funds and 16 public futures quote series. The NOAA download contains <b>{R['weekly_enso_rows']:,} weekly observations</b> from {R['weekly_enso_first']} to {R['weekly_enso_last']}, plus {R['oni_rows']:,} seasonal ONI records; only the relevant lagged overlap enters this study.</p>
<p>The earlier AGCL ambiguity is resolved by your edited configuration, which specifies <b>ACGL (Arch Capital)</b>. EG uses RE through 7 July 2023 and EG from 10 July 2023, with no duplicate dates. Massive's current EG dividend reference supplies the historical dividend series even though RE dividend queries returned zero. Full-month filters require at least 15 observations, an initial observation by day 7 and a final observation on or after day 25. Partial September 2026 is excluded from monthly outcomes. Missing prices are not forward-filled; daily returns spanning more than seven calendar days are rejected.</p>
{coverage_table}
<h3>Agricultural quote quality</h3><p>These vendor-designated futures histories do not provide a verified contract-by-contract roll ledger. We retain large positive/negative moves rather than select them away after seeing results. Zero reported volume does not necessarily invalidate a settlement-like price, so such prices remain in the descriptive sample. Cotton includes weekend placeholder rows that are removed before analysis. Some basket months drop out because all 16 constituents must be present.</p>{quality_table}
<div class="callout"><b>Scope gap:</b> the 16 series cover major grains, livestock and softs, not every agricultural future or exchange. No direct Massive futures history was obtained: catalogue, contract and price probes all returned 403. No claim of a completed contract-level futures execution backtest is made. Regional weather, catastrophe-loss records, insurer exposure maps and archived climate releases were outside this intentionally simple study.</div>
<h3>Source links</h3><ol class="sources">
<li><a href="https://massive.com/docs/rest/stocks/aggregates/custom-bars">Massive historical stock bars</a> and <a href="https://massive.com/docs/rest/stocks/corporate-actions/dividends">cash dividends</a>, <a href="https://massive.com/docs/rest/stocks/corporate-actions/splits">splits</a>. Authenticated responses are cached without API credentials.</li>
<li><a href="https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for">NOAA weekly SST/anomaly file</a>; <a href="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt">NOAA ONI file</a>; <a href="https://www.cpc.ncep.noaa.gov/data/indices/">index definitions and current methodology links</a>. This report uses the downloaded current files and does not assume the older ERSSTv5 labelling still applies to every updated series.</li>
<li><a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/">NOAA explanation of seasonal ONI and revisions</a>. A present-day archive does not reconstruct historical information availability.</li>
<li><a href="https://finance.yahoo.com/quote/ZC=F/history/">Yahoo corn futures history</a>, <a href="https://finance.yahoo.com/quote/ZS=F/history/">soybeans</a>, <a href="https://finance.yahoo.com/quote/ZW=F/history/">wheat</a>. The other symbols are listed in the coverage table; data were downloaded from Yahoo's public chart endpoint and cached, not from Massive.</li>
<li><a href="https://www.aoml.noaa.gov/how-does-el-nino-impact-atlantic-hurricane-season/">NOAA: ENSO and Atlantic hurricane activity</a>. This supports a physical risk channel, not the stock-price probability estimates.</li>
<li><a href="https://ir.archgroup.com/">Arch Capital issuer information</a>; <a href="https://www.sec.gov/Archives/edgar/data/1095073/000109507324000009/eg-20231231.htm">Everest's RE/EG ticker change</a>.</li>
<li><a href="https://www.invesco.com/content/dam/invesco/us/en/product-documents/etf/fact-sheet/dba-invesco-db-agriculture-fund-fact-sheet.pdf">DBA methodology</a>; <a href="https://teucrium.com/corn">CORN</a>; <a href="https://teucrium.com/agricultural-commodity-etfs">Teucrium agricultural funds</a>. Share histories embed their own rolling, fees and portfolio construction.</li></ol>
<h3>Reproduce the research</h3><p>All calculations run from local cached inputs. The accompanying folder includes raw JSON/text responses, normalised daily/monthly JSON Lines, calculated probabilities and model paths, SHA-256 checksums, and the downloader/analysis/report scripts. The report is self-contained and does not need an internet connection to display its charts. It contains no API key.</p><div class="formula">python3 enso_study_analyze.py<br>python3 enso_study_report.py<br>python3 -m unittest test_enso_study.py</div>
<p class="caption">Raw retrieval and results timestamps are stored in the files. The supplied API key is used only by the downloader via the separate local config.json; that configuration is deliberately excluded from the research delivery bundle.</p></section>
<footer>Research calculations are sample-dependent historical estimates. Association is not causal attribution. No trades were placed. Prepared from locally cached Massive, NOAA and Yahoo inputs.</footer></main>
<script>const scenarios={json.dumps(R['downside_probabilities'])};function scenario(){{const p=document.getElementById('phase').value,s=document.getElementById('season').value;document.getElementById('scenario-cards').innerHTML=scenarios.filter(x=>x.phase===p&&x.season===s).map(x=>`<div class="scenario"><span>${{x.asset}}</span><strong>${{(100*x.probability).toFixed(1)}}%</strong><small>${{x.negative_months}} declines / ${{x.months}} months</small><small>95% interval ${{(100*x.ci_low).toFixed(1)}}–${{(100*x.ci_high).toFixed(1)}}%</small><small>Baseline ${{(100*x.baseline).toFixed(1)}}%</small></div>`).join('')}}document.getElementById('phase').addEventListener('change',scenario);document.getElementById('season').addEventListener('change',scenario);scenario();</script></body></html>'''
    dest=ROOT/'ENSO_insurance_agriculture_research.html'
    dest.write_text(html_body)
    print(dest)
    print('HTML bytes',dest.stat().st_size)

if __name__=='__main__':main()
