"""Create a standalone scientific report and shareable PNG figures."""
from pathlib import Path
import base64, hashlib, html, json, os, platform
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/enso2026-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'calculated'; FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES={9:'September',10:'October',11:'November',12:'December'}
SOURCES={
 'CDS seasonal ensembles':'https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels',
 'ERA5 monthly fields':'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means',
 'Current NOAA ONI (ERSSTv6)':'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/',
 'NOAA monthly Niño indices':'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices',
 'NOAA state and division temperatures':'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/',
 'NOAA ENSO discussion, 10 September 2026':'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml',
 'CDS forecast and hindcast conventions':'https://confluence.ecmwf.int/spaces/CKB/pages/127318715/How+to+use+the+CDS+interactive+forms+and+CDS+API+for+seasonal+forecast+datasets',
 'ECMWF seasonal forecasting':'https://www.ecmwf.int/en/forecasts/documentation-and-support/seasonal',
}

def img(path):return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def fmt(v):return f'{v:+.1f}' if abs(v)>=.05 else '0.0'
def main():
    o=json.loads((OUT/'oni_forecast.json').read_text());s=json.loads((OUT/'state_enso_effects.json').read_text())
    df=pd.DataFrame(s['estimates']); f=pd.DataFrame(o['forecast']);obs=pd.read_json(OUT/'oni_v6.jsonl',lines=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    actual=obs[(obs.year==2026)&(obs.month<=7)]
    ax.plot(actual.month,actual.oni_c,'o-',color='#183c58',lw=2,label='Observed NOAA ONI')
    ax.errorbar(f.month,f.oni_c,yerr=np.array([f.oni_c-f.p10_c,f.p90_c-f.oni_c]),fmt='o-',capsize=5,lw=2,color='#ca5827',label='Forecast and nominal 80% interval')
    mp=pd.DataFrame(o['model_means'])
    for model in ['ecmwf','dwd']:
        ax.plot(mp.month,mp[model],':',lw=1.5,label=model.upper()+' calibrated comparison')
    ax.axhline(0,color='#cccccc',lw=1);ax.axhline(2,color='#c6a290',ls='--',lw=1)
    ax.set(xticks=range(1,13),xticklabels=['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ'],ylabel='ONI (°C)',title='2026 Pacific ENSO: observed ONI and calibrated forecast')
    ax.legend(loc='upper left',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(FIG/'oni_forecast.png',dpi=170);plt.close(fig)
    pivot=df.pivot(index='state',columns='month',values='enso_effect_c').sort_index()
    fig,ax=plt.subplots(figsize=(8,15),layout='constrained')
    im=ax.imshow(pivot.values,cmap='RdBu_r',vmin=-4,vmax=4,aspect='auto')
    ax.set(yticks=range(50),yticklabels=pivot.index,xticks=range(4),xticklabels=[NAMES[x] for x in pivot.columns],title='Estimated ENSO-associated monthly temperature contribution (°C)')
    ax.tick_params(axis='both',length=0,labelsize=9)
    for i,state in enumerate(pivot.index):
        for j,m in enumerate(pivot.columns):
            v=pivot.loc[state,m];row=df[(df.state==state)&(df.month==m)].iloc[0]
            label=fmt(v)+(' •' if row.effect_80pct_interval_excludes_zero else '')
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if abs(v)>2.4 else '#172833')
    cb=fig.colorbar(im,ax=ax,shrink=.45,pad=.03);cb.set_label('Difference from ENSO-neutral estimate (°C)')
    fig.set_layout_engine('constrained',rect=(0,.04,1,.96))
    fig.text(.03,.002,'• Nominal 80% effect interval excludes zero. Monthly estimates; not full weather anomalies.\nSeptember–October have no demonstrated added skill in recent conditional tests.',fontsize=8)
    fig.savefig(FIG/'state_enso_effects.png',dpi=170,bbox_inches='tight');plt.close(fig)
    # Independent checks displayed without obscuring forecast limitations.
    latest=actual.iloc[-1]; selected=o['selection']['model'].upper()
    modeltable=''.join(f'<tr><td>{k.upper()}</td><td>{v:.3f}</td></tr>' for k,v in o['validation_2017_2025_rmse_c'].items())
    onitable=''.join(f'<tr><td>{NAMES[r.month]}</td><td>{r.season}</td><td><b>{fmt(r.oni_c)}</b></td><td>{fmt(r.p10_c)} to {fmt(r.p90_c)}</td></tr>' for r in f.itertuples())
    statetable=''.join('<tr><th>'+state+'</th>'+''.join(f'<td>{fmt(pivot.loc[state,m])}</td>' for m in [9,10,11,12])+'</tr>' for state in pivot.index)
    detail=''.join(f'<tr><td>{r.state}</td><td>{NAMES[r.month]}</td><td>{fmt(r.enso_effect_c)}</td><td>{fmt(r.p10_c)} to {fmt(r.p90_c)}</td><td>{r.residual_weather_sd_c:.1f}</td><td>{r.n_years}</td></tr>' for r in df.sort_values(['state','month']).itertuples())
    valtable=''.join(f'<tr><td>{NAMES[int(m)]}</td><td>{r["conditional_rmse_c"]:.2f}</td><td>{r["trend_only_rmse_c"]:.2f}</td><td>{r["conditional_skill"]:+.1%}</td><td>{r["end_to_end_skill"]:+.1%}</td></tr>' for m,r in s['validation'].items())
    sourcehtml=''.join(f'<li><a href="{url}">{name}</a></li>' for name,url in SOURCES.items())
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ENSO outlook · September–December 2026</title>
<style>body{{max-width:1100px;margin:45px auto;padding:0 25px;font:16px/1.6 system-ui;color:#193044;background:#fff}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:45px;color:#184863}}h3{{margin-top:25px}}.meta{{color:#607789}}.note{{padding:18px 22px;border-left:4px solid #ca5827;background:#fff5ee}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:9px 12px;text-align:right;border-bottom:1px solid #dde5e9;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}thead{{background:#eaf1f5}}tbody tr:nth-child(even){{background:#f6f8fa}}img{{width:100%;height:auto}}a{{color:#125b87}}code{{background:#edf2f5;padding:2px 5px}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:30px}}@media(max-width:700px){{.columns{{display:block}}body{{padding:0 12px}}th,td{{padding:6px}}}}@media print{{body{{margin:0;font-size:11px}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style>
<p class="meta">RESEARCH FORECAST · AS OF 11 SEPTEMBER 2026 · ALL TEMPERATURE DIFFERENCES IN °C</p>
<h1>A very strong Pacific El Niño.<br>Uneven temperature effects across the United States.</h1>
<p>The calibrated {selected} seasonal ensemble projects ONI rising from <b>+2.6°C in ASO to +3.5°C in NDJ</b>. Historical state responses suggest a substantially stronger temperature signal in December than in early autumn. This analysis downloaded data directly from the Copernicus CDS API and checked forecasts against NOAA ONI observations.</p>
<div class="note"><b>What the state estimates mean.</b> These are statistical temperature contributions associated with ENSO, relative to ONI = 0 with the fitted trend held fixed. They do not establish the temperature change caused solely by ENSO, and they are not forecasts of the entire deviation from normal. September–October add no demonstrated skill over a trend-only model in the recent conditional evaluation. Every projected ONI value exceeds its season’s historical training maximum; the intervals cannot capture all risks of extrapolating to a record event.</div>
<h2>Pacific ONI forecast</h2>
<p>Latest observed ONI: <b>JJA 2026 = +1.8°C</b>, from NOAA’s current ERSSTv6 series. ONI is a three-month index: the month below is the <b>center month</b>. December therefore means November 2026–January 2027. The estimates are independent research forecasts, not official NOAA ONI forecasts. NOAA’s official ENSO forecast metric is now RONI; RONI probabilities have not been substituted for ONI values.</p>
<table><thead><tr><th>Center month</th><th>Season</th><th>Predicted ONI</th><th>Nominal 80% interval</th></tr></thead><tbody>{onitable}</tbody></table>
{img(FIG/'oni_forecast.png')}
<h2>Monthly ENSO contribution by state</h2>
<p>Positive values indicate warming relative to the fitted ENSO-neutral counterfactual; negative values indicate cooling. Multiply a Celsius difference by 1.8 for Fahrenheit. These are statewide monthly averages; local conditions can differ considerably. A value of 0.0 is a rounded estimate, not proof of no effect.</p>
{img(FIG/'state_enso_effects.png')}
<table><thead><tr><th>State</th><th>September</th><th>October</th><th>November</th><th>December</th></tr></thead><tbody>{statetable}</tbody></table>
<h2>Model choice and tests against real data</h2>
<p>Coupled seasonal ocean–atmosphere ensembles are appropriate for this horizon. The study compares ECMWF system 51, DWD system 22, CMCC system 4, their equally weighted calibrated combination, an observation-only model and persistence. It does not train a weather neural network from this small sample.</p>
<p>Model and ridge penalty selection used expanding-window tests from 2003–2016, with no future target values entering each fit. {selected} with penalty 0.1 ranked first in that selection period (RMSE 0.246°C). The selected specification was frozen before evaluation over 2017–2025, excluding 2024 because DWD returned no September data for that year. The final forecast refits that specification to the common 32-year sample from 1993–2025. ECMWF performed better in the later evaluation; retaining the original selection avoids choosing a winner after inspecting the test set.</p>
<div class="columns"><div><h3>ONI evaluation</h3><table><thead><tr><th>Model</th><th>ONI RMSE, °C</th></tr></thead><tbody>{modeltable}</tbody></table></div><div><h3>Interpretation</h3><p>The selected CMCC calibration reduced RMSE from persistence’s 0.566°C to 0.294°C across the four target seasons. The equally weighted combination scored 0.298°C and ECMWF 0.273°C. These are 32 errors across eight years, not 32 independent ENSO events.</p><p>This is a chronological retrospective evaluation using current-vintage observations and modern-system hindcasts. It is not a reconstruction of all information and model versions available to an actual historical forecaster.</p></div></div>
<h3>State temperature evaluation</h3>
<p>The conditional test below uses observed target-season ONI and forecasts temperature from prior years only, over 2011–2025. It tests the response relationship. The final column instead uses the selected historical ONI forecasts for the eight common years during 2017–2025. Skill is 1 − model MSE / trend-only MSE; negative values mean the ENSO model was worse. These aggregate results pool states equally; state errors are strongly correlated.</p>
<table><thead><tr><th>Month</th><th>Conditional RMSE</th><th>Trend-only RMSE</th><th>Conditional skill</th><th>Forecast-ONI skill</th></tr></thead><tbody>{valtable}</tbody></table>
<h2>Data collected and calculations</h2>
<ul><li><b>CDS forecasts:</b> September 2026 initialization; all 151 members across three systems; monthly SST for September–January on a 1° grid covering Niño-3.4 (5°S–5°N, 170°W–120°W). Matching September historical runs cover 1993–2025, with DWD 2024 missing. Individual member values and gridded fields are retained.</li>
<li><b>ERA5:</b> January–August 2026 monthly SST, 10 m zonal and meridional wind, and mean sea-level pressure at 0.25° across 10°S–10°N, 120°E–80°W. These are diagnostic inputs, not extra regressors trained on eight months. The coupled seasonal systems already encode atmosphere–ocean evolution; this study does not separately estimate a subsurface heat-content predictor. Recent ERA5 data can be provisional.</li>
<li><b>Observed ONI:</b> 919 ERSSTv6 seasonal values, 1950–JJA 2026. The older version is retained separately and is never spliced into the main series. NOAA OISST monthly regional indices are retained as a cross-check; differing SST analyses and baselines mean they need not equal official ONI.</li>
<li><b>Observed U.S. temperatures:</b> 77,488 monthly state values and 564,516 climate-division values, covering all 50 states and 369 divisions. They originate from NOAA’s September 2026 release. State modelling uses 1950–2025, or 1991–2025 for Hawaii. Fahrenheit observations are converted to Celsius; missing-value sentinels are removed. Division data provide a finer spatial archive, while estimates in this report use official state aggregates.</li></ul>
<p>Grid-cell SST is averaged using cosine-latitude weights. Each model’s raw seasonal SST and the observed JJA ONI are standardized using training data only and mapped to the target ERSSTv6 ONI using ridge regression with an intercept. For ASO, September–October forecast SST plus observed JJA ONI predict the ASO target; this is a calibrated bridge, not an assertion that two months constitute ONI. SON, OND and NDJ use the corresponding three forecast months. CDS lead month 1 is September for a September initialization. The January 2027 forecast is included to construct NDJ.</p>
<p>For each state and calendar month, the response model is <code>T = intercept + trend × (year − 1990)/10 + β × ONI</code>. Only β is penalized. The temperature contribution is <code>β × forecast ONI</code>, with the intercept and trend held fixed at the neutral counterfactual. A month-specific penalty is selected across states using 2000–2010 chronological tests and then frozen. September and November select 100; October and December select 0. Different penalties contribute to differences in estimated effect magnitudes across months.</p>
<h2>Uncertainty and limits</h2>
<p>ONI intervals use Student-t tails with seven degrees of freedom. Their scale is the larger of held-out RMSE or calibrated ensemble spread combined with model disagreement, inflated for calibration leverage. Three thousand correlated forecast draws preserve estimated dependence between overlapping seasons. These are nominal intervals, not proof of 80% coverage during an unprecedented event.</p>
<p>State intervals combine those ONI draws with 3,000 five-year moving-block bootstrap fits of each state’s historical temperature response. They cover uncertainty in the estimated ENSO-associated mean contribution. They <b>exclude unrelated weather variability</b>, such as blocking, cold-air outbreaks, soil-moisture effects and other circulation modes. The residual weather standard deviation is supplied separately below. Intervals are per estimate and have not been adjusted for comparisons across 200 state-month estimates. Hawaii has a shorter history.</p>
<p>Linear extrapolation, a changing climate and evolving ENSO teleconnections are material limitations. A calendar trend does not remove every confounder. Large December estimates in northern states should be treated as conditional scenarios, not deterministic claims that the state will be that much warmer than its 1991–2020 normal. The ENSO contribution is defined relative to ONI = 0, not relative to an arbitrary temperature climatology. September is already underway at the issue date; this forecast uses the September 1 seasonal initialization and complete observations through August, without assimilating September’s realized weather.</p>
<h2>All 200 estimates with uncertainty</h2>
<table><thead><tr><th>State</th><th>Month</th><th>ENSO effect, °C</th><th>80% effect interval, °C</th><th>Residual weather SD, °C</th><th>Years</th></tr></thead><tbody>{detail}</tbody></table>
<h2>Sources and reproducibility</h2><ul>{sourcehtml}</ul>
<p>Contains modified Copernicus Climate Change Service information (2026). Neither the European Commission nor ECMWF is responsible for subsequent use. NOAA observations and CDS products are credited to their originating institutions. Source NetCDF and text files, exact API requests, timestamps, hashes, calculated JSON records, code and validation results accompany this report. The archive excludes credentials.</p>
</html>'''
    (ROOT/'ENSO_forecast_September_December_2026.html').write_text(text)
    # A concise text counterpart provides the full state table without requiring a browser.
    md=['# ENSO forecast: September–December 2026','As of 11 September 2026. Research estimates; ONI uses centered three-month seasons.','',
    '| Center month | Season | ONI °C | Nominal 80% interval |','|---|---|---:|---:|']
    for r in f.itertuples():md.append(f'| {NAMES[r.month]} | {r.season} | {fmt(r.oni_c)} | {fmt(r.p10_c)} to {fmt(r.p90_c)} |')
    md+=['','Latest observed NOAA ERSSTv6 ONI: JJA 2026 +1.8°C. Official NOAA ENSO outlooks now use RONI, which is a different index.','',
    'State values below are ENSO-associated differences from a fitted ENSO-neutral counterfactual in °C, not total temperature departures from normal or proven causal effects. September–October have no demonstrated added skill in recent conditional tests. All ONI forecasts extrapolate beyond their seasonal training maxima. See the HTML report for all uncertainty intervals, methodology and validation.','',
    '| State | September | October | November | December |','|---|---:|---:|---:|---:|']
    for state in pivot.index:md.append('| '+state+' | '+' | '.join(fmt(pivot.loc[state,m]) for m in [9,10,11,12])+' |')
    md+=['','Selected CMCC calibration: ONI validation RMSE 0.294°C versus persistence 0.566°C, across eight held-out years. ECMWF comparison RMSE 0.273°C. Selection used 2003–2016; evaluation used 2017–2025, excluding 2024. Current-vintage data and modern hindcasts limit point-in-time interpretation.','','Sources:']
    md += [f'- [{k}]({v})' for k,v in SOURCES.items()]
    (ROOT/'forecast_summary.md').write_text('\n'.join(md)+'\n')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'raw').iterdir() if p.is_file()}
    (ROOT/'raw_sha256.json').write_text(json.dumps(hashes,indent=2))
    versions={'python':platform.python_version()}
    from importlib.metadata import version
    for name in ['numpy','pandas','scipy','xarray','cdsapi','netCDF4','matplotlib','beautifulsoup4']:versions[name]=version(name)
    (ROOT/'versions.json').write_text(json.dumps(versions,indent=2))
    print('Report and figures created',flush=True)

if __name__=='__main__':main()
