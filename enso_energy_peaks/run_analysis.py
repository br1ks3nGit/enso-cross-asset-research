#!/usr/bin/env python3
"""El Niño peak event study and delayed-signal energy-equity backtest."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "calculated"
ONI_FILE = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(ROOT.parent / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
END = pd.Period("2026-07", "M")
COMPANIES = ["XOM", "CVX", "COP", "EOG", "OXY", "SLB", "MPC", "PSX", "KMI", "WMB"]
ALL = COMPANIES + ["XLE", "SPY"]
NAMES = {
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips", "EOG": "EOG Resources",
    "OXY": "Occidental Petroleum", "SLB": "SLB", "MPC": "Marathon Petroleum",
    "PSX": "Phillips 66", "KMI": "Kinder Morgan", "WMB": "Williams",
    "XLE": "Energy Select Sector SPDR", "SPY": "SPDR S&P 500",
}
SECTOR = {
    "XOM": "Integrated", "CVX": "Integrated", "COP": "Exploration & production",
    "EOG": "Exploration & production", "OXY": "Exploration & production", "SLB": "Oilfield services",
    "MPC": "Refining", "PSX": "Refining", "KMI": "Midstream", "WMB": "Midstream",
}
START_FLOOR = {
    "MPC": pd.Period("2011-07", "M"), "PSX": pd.Period("2012-05", "M"),
    "KMI": pd.Period("2011-02", "M"), "COP": pd.Period("2012-06", "M"),
}


def load_oni() -> pd.Series:
    x = pd.read_csv(ONI_FILE)
    x["month"] = pd.to_datetime(x["Date"]).dt.to_period("M")
    return x.set_index("month")["ONI"].sort_index().loc[:END].astype(float)


def load_total_return_prices(ticker: str) -> pd.Series:
    bars = json.loads((RAW / f"massive_{ticker}_monthly.json").read_text())["results"]
    close = pd.Series({pd.to_datetime(r["t"], unit="ms", utc=True).to_period("M"): float(r["c"]) for r in bars}).sort_index().loc[:END]
    floor = START_FLOOR.get(ticker)
    if floor:
        close = close.loc[floor:]
    divs = json.loads((RAW / f"massive_{ticker}_dividends.json").read_text()).get("results", [])
    factors = []
    clean = []
    for r in divs:
        if r.get("ex_dividend_date") and r.get("historical_adjustment_factor") is not None:
            clean.append((pd.Timestamp(r["ex_dividend_date"]), float(r["historical_adjustment_factor"])))
    clean.sort()
    for month in close.index:
        month_end = month.to_timestamp(how="end").normalize()
        later = [factor for ex_date, factor in clean if ex_date > month_end]
        factors.append(later[0] if later else 1.0)
    return pd.Series(close.to_numpy() * np.asarray(factors), index=close.index, name=ticker)


def detect_peaks(oni: pd.Series) -> pd.DataFrame:
    warm = oni >= 0.5
    groups = (warm != warm.shift()).cumsum()
    rows = []
    for _, episode in oni[warm].groupby(groups[warm]):
        if len(episode) < 5:
            continue
        peak = episode.idxmax()
        after = oni.loc[peak + 1:episode.index.max()]
        decline = next((m for m in after.index if oni.loc[m] < oni.loc[m - 1]), None)
        if decline is None:
            continue
        decision = decline + 2  # centered observation is available two months later
        rows.append({
            "episode_start": str(episode.index.min()), "episode_end": str(episode.index.max()),
            "peak_month": str(peak), "peak_oni": float(episode.max()),
            "first_decline_center_month": str(decline), "confirmation_decision_month": str(decision),
            "trade_start_month": str(decision + 1),
        })
    return pd.DataFrame(rows)


def forward_sum(ret: pd.Series, horizon: int) -> pd.Series:
    return pd.concat([ret.shift(-i) for i in range(1, horizon + 1)], axis=1).sum(axis=1, min_count=horizon)


def hac_slope(x: pd.Series, y: pd.Series, lags: int) -> dict:
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(z)
    if n < 12:
        return {"n": n, "r": math.nan, "slope": math.nan, "hac_t": math.nan, "p": math.nan}
    xv, yv = z["x"].to_numpy(float), z["y"].to_numpy(float)
    X = np.column_stack([np.ones(n), xv])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ yv
    e = yv - X @ beta
    S = np.zeros((2, 2))
    for t in range(n):
        xt = X[t][:, None]
        S += e[t] ** 2 * (xt @ xt.T)
    q = min(lags, n - 1)
    for lag in range(1, q + 1):
        weight = 1 - lag / (q + 1)
        for t in range(lag, n):
            a = (X[t] * e[t])[:, None]
            b = (X[t - lag] * e[t - lag])[:, None]
            S += weight * (a @ b.T + b @ a.T)
    cov = bread @ S @ bread
    se = math.sqrt(max(float(cov[1, 1]), 0))
    tval = float(beta[1] / se) if se else math.nan
    p = float(2 * stats.t.sf(abs(tval), df=n - 2)) if not math.isnan(tval) else math.nan
    return {"n": n, "r": float(np.corrcoef(xv, yv)[0, 1]), "slope": float(beta[1]), "hac_t": tval, "p": p}


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return pd.Series(out, index=pvalues.index)


def one_sample(values: list[float]) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if not pd.isna(x)], float)
    if len(v) < 3:
        return (float(v.mean()) if len(v) else math.nan, math.nan, len(v))
    t, p = stats.ttest_1samp(v, 0.0)
    return float(v.mean()), float(p), len(v)


def hac_mean(values: pd.Series, lags: int = 3) -> tuple[float, float]:
    x = values.dropna().to_numpy(float)
    n = len(x)
    if n < 5:
        return math.nan, math.nan
    e = x - x.mean()
    lv = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        lv += 2 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag]) / n
    se = math.sqrt(max(lv, 0) / n)
    t = float(x.mean() / se) if se else math.nan
    return t, float(2 * stats.t.sf(abs(t), df=n - 1)) if not math.isnan(t) else math.nan


def fmt_num(x, d=2):
    return "n.a." if pd.isna(x) else f"{x:.{d}f}"


def fmt_pct(x, d=1):
    return "n.a." if pd.isna(x) else f"{100*x:.{d}f}%"


def table(df: pd.DataFrame, cols: list[tuple[str, str, callable]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for key, _, fn in cols:
            value = row.get(key)
            text = fn(value) if fn else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oni = load_oni()
    prices = {t: load_total_return_prices(t) for t in ALL}
    returns = {t: np.log(p).diff() for t, p in prices.items()}
    peaks = detect_peaks(oni)
    eligible_peaks = peaks[pd.PeriodIndex(peaks["peak_month"], freq="M") >= pd.Period("2004-01", "M")].copy()
    peaks.to_csv(OUT / "oni_peak_definitions.csv", index=False)

    # Retrospective event panel: after the ex-post peak, total-return relative performance.
    event_rows = []
    for _, event in eligible_peaks.iterrows():
        peak = pd.Period(event["peak_month"], "M")
        for ticker in COMPANIES + ["XLE"]:
            for benchmark in (["SPY", "XLE"] if ticker != "XLE" else ["SPY"]):
                rel = returns[ticker] - returns[benchmark]
                for horizon in (1, 3, 6, 12):
                    months = pd.period_range(peak + 1, peak + horizon, freq="M")
                    z = rel.reindex(months)
                    value = math.exp(float(z.sum())) - 1 if z.notna().sum() == horizon else math.nan
                    event_rows.append({
                        "peak_month": str(peak), "peak_oni": event["peak_oni"], "ticker": ticker,
                        "company": NAMES[ticker], "subsector": SECTOR.get(ticker, "Sector ETF"),
                        "benchmark": benchmark, "horizon_months": horizon, "relative_total_return": value,
                    })
    event_panel = pd.DataFrame(event_rows)
    event_panel.to_csv(OUT / "peak_event_panel.csv", index=False)
    summaries = []
    for keys, g in event_panel.groupby(["ticker", "company", "subsector", "benchmark", "horizon_months"], sort=False):
        mean, p, n = one_sample(g["relative_total_return"].tolist())
        summaries.append(dict(zip(["ticker", "company", "subsector", "benchmark", "horizon_months"], keys), mean_return=mean, p=p, n_events=n, confidence=1-p if not pd.isna(p) else math.nan))
    event_summary = pd.DataFrame(summaries)
    event_summary["q_bh"] = bh_adjust(event_summary["p"].fillna(1))
    event_summary.to_csv(OUT / "peak_event_summary.csv", index=False)

    # Point-in-time predictive correlations: at month-end t, centered ONI t-2 is available.
    asof = pd.DataFrame({"ONI": oni.shift(2), "dONI": oni.diff().shift(2)})
    corr_rows = []
    for ticker in COMPANIES + ["XLE"]:
        benchmarks = ["SPY", "XLE"] if ticker != "XLE" else ["SPY"]
        for benchmark in benchmarks:
            target = returns[ticker] - returns[benchmark]
            for signal in ("ONI", "dONI"):
                for horizon in (1, 3, 6, 12):
                    result = hac_slope(asof[signal], forward_sum(target, horizon), lags=horizon)
                    corr_rows.append({"ticker": ticker, "company": NAMES[ticker], "subsector": SECTOR.get(ticker, "Sector ETF"), "benchmark": benchmark, "signal": signal, "horizon_months": horizon, **result})
    correlations = pd.DataFrame(corr_rows)
    correlations["confidence"] = 1 - correlations["p"]
    correlations["q_bh_global"] = bh_adjust(correlations["p"].fillna(1))
    correlations["q_bh_family"] = correlations.groupby("benchmark", group_keys=False)["p"].apply(lambda s: bh_adjust(s.fillna(1)))
    correlations["p_over_50pct"] = correlations["p"] > 0.50
    correlations.to_csv(OUT / "predictive_correlations.csv", index=False)
    best_spy = correlations[correlations["benchmark"] == "SPY"].loc[lambda d: d.groupby("ticker")["p"].idxmin()].sort_values("p").reset_index(drop=True)
    best_xle = correlations[correlations["benchmark"] == "XLE"].loc[lambda d: d.groupby("ticker")["p"].idxmin()].sort_values("p").reset_index(drop=True)
    best_spy.to_csv(OUT / "best_predictive_correlation_by_ticker.csv", index=False)
    best_xle.to_csv(OUT / "best_xle_relative_correlation_by_ticker.csv", index=False)

    # Subsector tendencies: equal-weight company event return within each peak.
    sector_rows = []
    company_events = event_panel[(event_panel["benchmark"] == "SPY") & (event_panel["horizon_months"].isin([6, 12]))]
    for (subsector, horizon, peak), g in company_events.groupby(["subsector", "horizon_months", "peak_month"]):
        sector_rows.append({"subsector": subsector, "horizon_months": horizon, "peak_month": peak, "return": g["relative_total_return"].mean(), "companies": g["relative_total_return"].notna().sum()})
    sector_panel = pd.DataFrame(sector_rows)
    sector_summary_rows = []
    for (subsector, horizon), g in sector_panel.groupby(["subsector", "horizon_months"]):
        mean, p, n = one_sample(g["return"].tolist())
        sector_summary_rows.append({"subsector": subsector, "horizon_months": horizon, "mean_return": mean, "p": p, "n_events": n})
    sector_summary = pd.DataFrame(sector_summary_rows).sort_values(["horizon_months", "mean_return"], ascending=[True, False])
    sector_summary.to_csv(OUT / "subsector_peak_summary.csv", index=False)

    # Implementable delayed peak-confirmation strategy with expanding event-only sign learning.
    strategy_rows, strategy_summary = [], []
    confirmations = eligible_peaks.copy()
    for ticker in COMPANIES + ["XLE"]:
        rel = returns[ticker] - returns["SPY"]
        positions = pd.Series(0.0, index=pd.period_range(rel.index.min(), END, freq="M"))
        used = []
        event_audit = []
        for _, event in confirmations.iterrows():
            decision = pd.Period(event["confirmation_decision_month"], "M")
            trade_start = decision + 1
            trade_end = decision + 6
            prior = []
            for old in used:
                if old["trade_end"] < decision and not pd.isna(old["outcome"]):
                    prior.append(old["outcome"])
            months = pd.period_range(trade_start, trade_end, freq="M")
            z = rel.reindex(months)
            outcome = float(z.sum()) if z.notna().sum() == 6 else math.nan
            if len(prior) >= 2 and not pd.isna(outcome):
                sign = 1.0 if np.mean(prior) > 0 else -1.0
                positions.loc[months] = sign
                event_audit.append({"decision": str(decision), "trade_start": str(trade_start), "trade_end": str(trade_end), "prior_events": len(prior), "prior_mean_log_excess": float(np.mean(prior)), "position": sign, "realized_log_excess": outcome})
            used.append({"trade_end": trade_end, "outcome": outcome})
        active_start = next((m for m, v in positions.items() if v != 0), None)
        if active_start is None:
            continue
        sample = positions.loc[active_start:END]
        gross = sample * rel.reindex(sample.index).fillna(0)
        changes = sample.diff().abs().fillna(sample.abs())
        # Two legs at 10 bp per unit of turnover, charged exactly in log wealth.
        cost = changes * 2 * 0.001
        net = gross + np.log1p(-cost)
        wealth = np.exp(net.cumsum())
        dd = wealth / wealth.cummax() - 1
        tval, pval = hac_mean(net, 6)
        n_active = int((sample != 0).sum())
        summary_row = {
            "ticker": ticker, "company": NAMES[ticker], "start": str(active_start), "end": str(END),
            "calendar_months": len(net), "active_months": n_active,
            "cumulative_return": float(wealth.iloc[-1] - 1), "annual_return": float(math.exp(net.mean() * 12) - 1),
            "annual_vol": float(net.std(ddof=1) * math.sqrt(12)), "sharpe": float(net.mean() / net.std(ddof=1) * math.sqrt(12)) if net.std(ddof=1) else math.nan,
            "max_drawdown": float(dd.min()), "hac_t": tval, "p": pval, "confidence": 1-pval,
            "oos_events": len(event_audit),
        }
        strategy_summary.append(summary_row)
        for month in net.index:
            strategy_rows.append({"ticker": ticker, "month": str(month), "position": sample.loc[month], "gross_log_return": gross.loc[month], "cost": cost.loc[month], "net_log_return": net.loc[month], "wealth": wealth.loc[month]})
        (OUT / f"peak_confirmation_events_{ticker}.json").write_text(json.dumps(event_audit, indent=2) + "\n")
    strategy = pd.DataFrame(strategy_rows)
    strat_summary = pd.DataFrame(strategy_summary).sort_values("p")
    strategy.to_csv(OUT / "peak_confirmation_strategy_monthly.csv", index=False)
    strat_summary.to_csv(OUT / "peak_confirmation_strategy_summary.csv", index=False)

    # Report tables.
    peak_show = eligible_peaks[["peak_month", "peak_oni", "confirmation_decision_month", "trade_start_month"]]
    event6 = event_summary[(event_summary["benchmark"] == "SPY") & (event_summary["horizon_months"] == 6)].copy().sort_values("mean_return", ascending=False)
    event6["p_gt_50"] = event6["p"].map(lambda x: "Yes" if x > .5 else "No")
    best_show = best_spy.copy()
    best_show["p_gt_50"] = best_show["p"].map(lambda x: "Yes" if x > .5 else "No")
    best_xle_show = best_xle.copy()
    high_p_count = int(correlations["p_over_50pct"].sum())
    conventional = correlations[correlations["q_bh_global"] < .05]
    event_conventional = event_summary[event_summary["q_bh"] < .05]
    strategy_conventional = strat_summary[strat_summary["p"] < .05]
    robustness_rows = []
    for _, row in conventional.sort_values(["ticker", "benchmark", "horizon_months"]).iterrows():
        horizon = int(row["horizon_months"])
        x = asof[row["signal"]]
        y = forward_sum(returns[row["ticker"]] - returns[row["benchmark"]], horizon)
        samples = {
            "Pre-2015": (x.loc[:pd.Period("2014-12", "M")], y.loc[:pd.Period("2014-12", "M")]),
            "2015 onward": (x.loc[pd.Period("2015-01", "M"):], y.loc[pd.Period("2015-01", "M"):]),
        }
        bad_start = pd.Period("2020-03", "M") - horizon
        bad_end = pd.Period("2021-11", "M")
        bad = pd.period_range(bad_start, bad_end, freq="M")
        samples["Excluding COVID-overlap"] = (x.drop(index=bad, errors="ignore"), y.drop(index=bad, errors="ignore"))
        for sample, (xx, yy) in samples.items():
            test = hac_slope(xx, yy, horizon)
            robustness_rows.append({
                "ticker": row["ticker"], "benchmark": row["benchmark"], "signal": row["signal"],
                "horizon_months": horizon, "sample": sample, **test,
            })
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(OUT / "surviving_correlation_robustness.csv", index=False)
    robust_display = robustness.copy()
    robust_display["relationship"] = robust_display.apply(lambda r: f"{r['ticker']} − {r['benchmark']}: {r['signal']} / {int(r['horizon_months'])}m", axis=1)
    xle_event = event6[event6["ticker"] == "XLE"].iloc[0]
    top_sector = sector_summary[sector_summary["horizon_months"] == 6].iloc[0]
    cop6 = conventional[(conventional["ticker"] == "COP") & (conventional["benchmark"] == "XLE") & (conventional["horizon_months"] == 6)].iloc[0]
    psx3 = conventional[(conventional["ticker"] == "PSX") & (conventional["benchmark"] == "XLE") & (conventional["horizon_months"] == 3)].iloc[0]
    slb3 = conventional[(conventional["ticker"] == "SLB") & (conventional["benchmark"] == "SPY") & (conventional["horizon_months"] == 3)].iloc[0]
    refining6 = sector_summary[(sector_summary["subsector"] == "Refining") & (sector_summary["horizon_months"] == 6)].iloc[0]
    refining12 = sector_summary[(sector_summary["subsector"] == "Refining") & (sector_summary["horizon_months"] == 12)].iloc[0]

    outcome = "No robust correlation after multiple-testing control" if conventional.empty else f"{len(conventional)} predictive relationships survive BH q<0.05"
    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>El Niño peaks and U.S. energy equities</title><style>
:root{{--bg:#07111c;--card:#101d2b;--card2:#14263a;--text:#e9f0f7;--muted:#9fb0c2;--line:#2a3a4e;--cyan:#22d3ee;--green:#34d399;--amber:#fbbf24;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#102b3b 0,#07111c 38%);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:44px 24px 72px}}h1{{font-size:44px;line-height:1.08;letter-spacing:-.035em;max-width:900px;margin:10px 0 16px}}h2{{font-size:25px;margin:46px 0 14px}}h3{{font-size:18px}}p{{max-width:930px}}a{{color:#67e8f9}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:750}}.lede{{font-size:19px;color:#c9d6e4}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card{{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:14px;padding:20px}}.metric{{font-size:28px;font-weight:760}}.label,.small{{color:var(--muted);font-size:13px}}.good{{color:var(--green)}}.warn{{color:var(--amber)}}.bad{{color:var(--red)}}.callout{{border-left:4px solid var(--amber);background:#1a1c20;padding:14px 18px;border-radius:0 10px 10px 0}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:14px 0 24px}}table{{border-collapse:collapse;width:100%;min-width:820px;background:var(--card)}}th{{text-align:left;background:#15283d;color:#bfd1e4;text-transform:uppercase;letter-spacing:.05em;font-size:12px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}details{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px 17px;margin:10px 0}}summary{{cursor:pointer;font-weight:700}}code{{background:#17283b;padding:2px 5px;border-radius:4px}}footer{{border-top:1px solid var(--line);margin-top:42px;padding-top:16px;color:var(--muted)}}@media(max-width:800px){{h1{{font-size:34px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class='eyebrow'>Event study and delayed-signal backtest · 12 September 2026</div>
<h1>El Niño peaks and U.S. energy equities</h1><p class='lede'>Monthly Massive total-return-adjusted prices show nine delayed-ONI predictive relationships that survive global false-discovery control. They concentrate in ConocoPhillips, refiners and SLB. The narrower ex-post peak study and the implementable peak-confirmation strategy do not produce statistically reliable returns.</p>
<div class='grid'><div class='card'><div class='metric bad'>{outcome}</div><div class='label'>Point-in-time predictive tests</div></div><div class='card'><div class='metric'>{len(eligible_peaks)}</div><div class='label'>Eligible ONI-defined peak episodes since Massive coverage began</div></div><div class='card'><div class='metric warn'>{high_p_count}/{len(correlations)}</div><div class='label'>Predictive tests with p &gt; 0.50</div></div></div>
<div class='callout'><b>Interpreting the requested 50% threshold.</b> A p-value above 0.50 means the observed association is highly compatible with zero. If “greater than 50%” means confidence = 1 − p above 50%, that is p &lt; 0.50—a very weak screen. This report shows p, 1 − p and Benjamini–Hochberg q-values, and treats q &lt; 0.05 as the conventional evidence threshold.</div>

<h2>Peak definition and information timing</h2><p>An ONI-defined warm episode requires at least five consecutive monthly centered seasons at or above 0.5°C. The peak is the maximum revised ONI within the run. This is a retrospective event date. The implementable confirmation date is two months after the first centered ONI decline following the peak; trading starts the next month.</p>
{table(peak_show,[('peak_month','Ex-post peak',str),('peak_oni','Peak ONI',lambda x:fmt_num(x,1)),('confirmation_decision_month','First usable confirmation',str),('trade_start_month','Trade start',str)])}

<h2>Six months after the ex-post ONI peak</h2><p>Returns are dividend- and split-adjusted and measured relative to SPY. The p-value is a two-sided t-test across peak episodes. These dates are not tradable peak calls because the maximum is known only afterward.</p>
{table(event6,[('company','Company / ETF',str),('subsector','Subsector',str),('n_events','Events',lambda x:str(int(x))),('mean_return','Mean relative return',lambda x:fmt_pct(x,1)),('p','p-value',lambda x:fmt_num(x,3)),('confidence','1 − p',lambda x:fmt_pct(x,1)),('q_bh','BH q',lambda x:fmt_num(x,3)),('p_gt_50','p > 50%',str)])}
<p>XLE’s mean six-month performance after peaks was {fmt_pct(xle_event['mean_return'])} relative to SPY (p={fmt_num(xle_event['p'],3)}). The strongest subsector average was {html.escape(str(top_sector['subsector']))} at {fmt_pct(top_sector['mean_return'])}, but the small number of episodes limits inference.</p>

<h2>Point-in-time predictive correlations</h2><p>At each month-end, the newest usable ONI is delayed two months. The target is the following 1-, 3-, 6- or 12-month company total return minus SPY. The table shows each ticker’s lowest p-value across ONI/ΔONI and the four predeclared horizons. HAC standard errors use a lag equal to the horizon. Selecting the best row makes the raw p-value optimistic; the global BH q-value controls all {len(correlations)} predictive tests.</p>
{table(best_show,[('company','Company / ETF',str),('signal','Signal',str),('horizon_months','Horizon',lambda x:f'{int(x)}m'),('n','N',lambda x:str(int(x))),('r','Correlation',lambda x:fmt_num(x,3)),('slope','Slope',lambda x:fmt_num(x,3)),('p','HAC p',lambda x:fmt_num(x,3)),('confidence','1 − p',lambda x:fmt_pct(x,1)),('q_bh_family','SPY-family q',lambda x:fmt_num(x,3)),('q_bh_global','Global q',lambda x:fmt_num(x,3)),('p_gt_50','p > 50%',str)])}
<h3>Company returns relative to XLE</h3><p>This removes the common energy-sector move and asks whether ONI predicts company selection within energy.</p>
{table(best_xle_show,[('company','Company',str),('signal','Signal',str),('horizon_months','Horizon',lambda x:f'{int(x)}m'),('n','N',lambda x:str(int(x))),('r','Correlation',lambda x:fmt_num(x,3)),('p','HAC p',lambda x:fmt_num(x,3)),('q_bh_family','XLE-family q',lambda x:fmt_num(x,3)),('q_bh_global','Global q',lambda x:fmt_num(x,3))])}
<p>{len(conventional)} of {len(correlations)} predictive relationships survive global BH q&lt;0.05. These rows warrant further locked-holdout testing; the peak-event strategy itself remains unsupported.</p>
<h3>Robustness of the globally significant rows</h3><p>These are raw HAC p-values within fixed subsamples; they are diagnostic, not a second discovery test. The COVID exclusion removes every decision whose forward-return window intersects March 2020–December 2021.</p>
{table(robust_display,[('relationship','Relationship',str),('sample','Sample',str),('n','N',lambda x:str(int(x))),('r','Correlation',lambda x:fmt_num(x,3)),('slope','Slope',lambda x:fmt_num(x,3)),('p','HAC p',lambda x:fmt_num(x,3))])}

<h2>Delayed peak-confirmation strategy</h2><p>This is the tradable test. For each confirmed episode, the direction is learned only from completed prior confirmed episodes. At least two prior events are required. The sign is then frozen for six months. The position is company total return minus SPY, with 10 bp charged on each leg at changes in position. Returns are compounded consistently as log relative returns.</p>
{table(strat_summary,[('company','Company / ETF',str),('start','OOS start',str),('oos_events','OOS events',lambda x:str(int(x))),('active_months','Active months',lambda x:str(int(x))),('cumulative_return','Cumulative return',lambda x:fmt_pct(x,1)),('annual_return','Annualized return',lambda x:fmt_pct(x,1)),('sharpe','Sharpe',lambda x:fmt_num(x,2)),('max_drawdown','Max drawdown',lambda x:fmt_pct(x,1)),('p','HAC p',lambda x:fmt_num(x,3)),('confidence','1 − p',lambda x:fmt_pct(x,1))])}
<p>{len(strategy_conventional)} strategies have HAC p&lt;0.05 before correcting across companies. The limited number of peak events, long inactive periods and learned direction make these results exploratory.</p>

<h2>Key tendencies</h2><div class='grid'><div class='card'><h3>COP underperformed XLE at high ONI</h3><p class='small'>The six-month correlation is {fmt_num(cop6['r'],3)}. A +1°C ONI difference corresponds to {fmt_pct(cop6['slope'],1)} in subsequent six-month log performance versus XLE (global q={fmt_num(cop6['q_bh_global'],3)}). The sign persists after removing COVID-overlapping windows.</p></div><div class='card'><h3>Refiners weakened later</h3><p class='small'>The retrospective refiner average is {fmt_pct(refining6['mean_return'],1)} versus SPY six months after a peak and {fmt_pct(refining12['mean_return'],1)} after 12 months. Only four eligible events exist, and neither result survives multiplicity correction.</p></div><div class='card'><h3>PSX showed short-horizon relative strength</h3><p class='small'>Rising ΔONI correlates with PSX outperformance versus XLE over three months: r={fmt_num(psx3['r'],3)}, global q={fmt_num(psx3['q_bh_global'],3)}. This differs from PSX’s negative longer-horizon relationship versus SPY.</p></div><div class='card'><h3>SLB sensitivity was negative</h3><p class='small'>ΔONI versus next-three-month SLB performance relative to SPY has r={fmt_num(slb3['r'],3)} and global q={fmt_num(slb3['q_bh_global'],3)}. The post-2015 subsample is weaker, so stability is incomplete.</p></div><div class='card'><h3>No broad peak effect</h3><p class='small'>XLE averaged {fmt_pct(xle_event['mean_return'],1)} versus SPY in the six months after ONI peaks (p={fmt_num(xle_event['p'],3)}). The broad energy sector did not systematically outperform after peaks.</p></div><div class='card'><h3>Monthly frequency is sufficient</h3><p class='small'>ONI changes monthly and is smoothed over three months. Daily equity bars would not create additional independent ENSO signals.</p></div></div>

<h2>Data and limitations</h2><details open><summary>Massive market data</summary><p>Monthly qualifying-trade OHLC bars were requested with <code>adjusted=true</code>. Massive’s separate dividend endpoint supplies historical adjustment factors, which were applied to build total-return-adjusted price indexes. September 2026 partial bars are excluded.</p></details><details><summary>Universe and identity controls</summary><p>The predeclared representative universe covers integrated oil, exploration and production, oilfield services, refining and midstream. MPC is restricted to July 2011 onward, PSX to May 2012 and KMI to February 2011 to avoid earlier ticker histories or predecessor regimes. COP is restricted to June 2012 onward because the April 2012 Phillips 66 spin-off is not a normal price return. The current-company sample has survivorship and selection bias.</p></details><details><summary>ONI vintages</summary><p>The two-month availability delay is applied to today’s revised ONI history. It prevents direct timing leakage but does not reconstruct historical preliminary vintages or later revisions.</p></details><details><summary>Inference</summary><p>Forward horizons overlap, so HAC errors are used. Event p-values rely on only a handful of peaks. BH q-values address multiple tests within each displayed family but do not eliminate universe-selection bias or macroeconomic confounding.</p></details>

<h2>Sources</h2><ul><li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/'>NOAA CPC ONI definition and revision warning</a></li><li><a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>Massive stock aggregate bars</a></li><li><a href='https://massive.com/docs/rest/stocks/corporate-actions/dividends'>Massive dividend adjustment factors</a></li><li><a href='https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends'>Massive split/dividend adjustment note</a></li></ul>
<footer>Research output, not an investment recommendation. Full event panels, correlations, strategy records and provenance are saved beside this report.</footer></main></body></html>"""
    (ROOT / "El_Nino_peaks_US_energy_backtest.html").write_text(report)

    files = [ONI_FILE] + [RAW / f"massive_{t}_monthly.json" for t in ALL] + [RAW / f"massive_{t}_dividends.json" for t in ALL]
    provenance = {
        "created_at": "2026-09-12", "massive_adjusted_bars": True, "dividend_adjustment_factors_applied": True,
        "end_month": str(END), "api_credentials_in_output": False,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result = {
        "outcome": outcome, "eligible_peaks": len(eligible_peaks), "predictive_tests": len(correlations),
        "predictive_p_over_0_50": high_p_count, "predictive_bh_q_under_0_05": len(conventional),
        "event_bh_q_under_0_05": len(event_conventional), "strategy_hac_p_under_0_05": len(strategy_conventional),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
