#!/usr/bin/env python3
"""Leakage-controlled ENSO/metals/cooling-infrastructure walk-forward study."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "calculated"
INPUT = Path(os.environ.get(
    "ENSO_METALS_INPUT_DIR",
    str(ROOT.parent / "inputs" / "enso_oni_metals"),
))
END_MONTH = pd.Period("2026-07", "M")  # last complete month common to supplied ONI data
METALS = ["Copper", "Aluminum", "Nickel"]
EQUITIES = ["MOD", "VRT", "TT", "CARR", "JCI", "ETN"]
PURE_PLAY = {"MOD", "VRT", "TT", "CARR"}
NAMES = {
    "MOD": "Modine",
    "VRT": "Vertiv",
    "TT": "Trane Technologies",
    "CARR": "Carrier",
    "JCI": "Johnson Controls",
    "ETN": "Eaton",
}


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 4 or z.iloc[:, 0].std() == 0 or z.iloc[:, 1].std() == 0:
        return math.nan, math.nan, len(z)
    r, p = stats.pearsonr(z.iloc[:, 0], z.iloc[:, 1])
    return float(r), float(p), len(z)


def hac_mean_test(r: pd.Series, lags: int = 3) -> tuple[float, float]:
    """Newey-West t-test for mean with Bartlett kernel."""
    x = r.dropna().to_numpy(float)
    n = len(x)
    if n < 5:
        return math.nan, math.nan
    e = x - x.mean()
    long_var = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(e[lag:] @ e[:-lag]) / n
        long_var += 2 * (1 - lag / (lags + 1)) * gamma
    se = math.sqrt(max(long_var, 0) / n)
    if se == 0:
        return math.nan, math.nan
    t = float(x.mean() / se)
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return t, p


def strategy_metrics(ret: pd.Series, actual: pd.Series, pred: pd.Series) -> dict:
    r = ret.dropna()
    n = len(r)
    if n == 0:
        return {k: math.nan for k in ("n", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "hac_t", "hac_p", "ic", "ic_p")}
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    ann_return = wealth.iloc[-1] ** (12 / n) - 1 if (wealth > 0).all() else math.nan
    ann_vol = r.std(ddof=1) * math.sqrt(12)
    t, p = hac_mean_test(r)
    ic, ic_p, _ = pearson(pred, actual)
    return {
        "n": n,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if r.std(ddof=1) else math.nan,
        "max_drawdown": float(dd.min()),
        "hit_rate": float((r > 0).mean()),
        "hac_t": t,
        "hac_p": p,
        "ic": ic,
        "ic_p": ic_p,
    }


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    base = pd.read_csv(INPUT / "aligned_monthly_ONI_metals_1992_2026-07.csv")
    base["month"] = pd.to_datetime(base["Date"]).dt.to_period("M")
    base = base.set_index("month").sort_index().loc[:END_MONTH]
    for m in METALS:
        base[f"r_{m}"] = np.log(base[m]).diff()
    equities = {}
    for ticker in EQUITIES + ["SPY"]:
        payload = json.loads((RAW / f"massive_{ticker}_monthly.json").read_text())
        rows = payload.get("results", [])
        s = pd.Series(
            {pd.to_datetime(row["t"], unit="ms", utc=True).to_period("M"): float(row["c"]) for row in rows},
            name=ticker,
        ).sort_index().loc[:END_MONTH]
        # Avoid historical symbol reuse/discontinuity for post-2020 HVAC issuers.
        if ticker == "VRT":
            s = s.loc[pd.Period("2020-02", "M"):]
        elif ticker == "TT":
            s = s.loc[pd.Period("2020-03", "M"):]
        elif ticker == "CARR":
            s = s.loc[pd.Period("2020-04", "M"):]
        equities[ticker] = np.log(s).diff()
    return base, equities


def oni_candidates(base: pd.DataFrame, publication_delay: int = 2) -> dict[str, pd.Series]:
    """Predict target month t at end of t-1. Latest centered ONI is t-(delay+1)."""
    out = {}
    for signal, source in (("ONI", base["ONI"]), ("dONI", base["ONI"].diff())):
        for extra_lag in range(13):
            shift = publication_delay + 1 + extra_lag
            out[f"{signal} lag {extra_lag}"] = source.shift(shift)
    return out


def metal_candidates(base: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for metal in METALS:
        for lag in range(1, 7):
            out[f"{metal} return lag {lag}"] = base[f"r_{metal}"].shift(lag)
    return out


def walk_forward(
    y: pd.Series,
    candidates: dict[str, pd.Series],
    min_train: int,
    test_months: int = 12,
    cost_bps: float = 10,
    legs: int = 1,
) -> pd.DataFrame:
    index = y.dropna().index.intersection(pd.Index(sorted(set().union(*(s.dropna().index for s in candidates.values())))))
    if len(index) <= min_train:
        return pd.DataFrame()
    first_test = index[min_train]
    last = index[-1]
    rows = []
    test_start = first_test
    previous_position = 0.0
    while test_start <= last:
        test_end = min(test_start + (test_months - 1), last)
        train_end = test_start - 1
        best = None
        for name, x in candidates.items():
            z = pd.concat([x.rename("x"), y.rename("y")], axis=1).loc[:train_end].dropna()
            if len(z) < min_train:
                continue
            r, p = stats.pearsonr(z["x"], z["y"])
            candidate = (abs(float(r)), name, float(r), float(p), z)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            test_start = test_end + 1
            continue
        _, name, train_r, train_p, z = best
        x = candidates[name]
        x_mean, y_mean = z["x"].mean(), z["y"].mean()
        beta = float(((z["x"] - x_mean) * (z["y"] - y_mean)).sum() / ((z["x"] - x_mean) ** 2).sum())
        alpha = float(y_mean - beta * x_mean)
        for month in pd.period_range(test_start, test_end, freq="M"):
            if month not in y.index or month not in x.index or pd.isna(y.get(month)) or pd.isna(x.get(month)):
                continue
            prediction = alpha + beta * float(x.loc[month])
            position = 1.0 if prediction > 0 else -1.0 if prediction < 0 else 0.0
            turnover = abs(position - previous_position)
            net = position * float(y.loc[month]) - turnover * legs * cost_bps / 10000
            rows.append({
                "month": str(month), "train_end": str(train_end), "test_end": str(test_end),
                "spec": name, "train_n": len(z), "train_r": train_r, "train_p": train_p,
                "alpha": alpha, "beta": beta, "x": float(x.loc[month]), "prediction": prediction,
                "actual": float(y.loc[month]), "position": position, "turnover": turnover,
                "strategy_net": net,
            })
            previous_position = position
        test_start = test_end + 1
    return pd.DataFrame(rows)


def descriptive_best(y: pd.Series, candidates: dict[str, pd.Series]) -> dict:
    rows = []
    for name, x in candidates.items():
        r, p, n = pearson(x, y)
        rows.append({"spec": name, "r": r, "p": p, "n": n})
    return max(rows, key=lambda a: abs(a["r"]) if not math.isnan(a["r"]) else -1)


def summarize_wf(label: str, wf: pd.DataFrame, target_kind: str) -> dict:
    if wf.empty:
        return {"asset": label, "target_kind": target_kind, "n": 0}
    idx = pd.PeriodIndex(wf["month"], freq="M")
    actual = pd.Series(wf["actual"].to_numpy(), index=idx)
    pred = pd.Series(wf["prediction"].to_numpy(), index=idx)
    ret = pd.Series(wf["strategy_net"].to_numpy(), index=idx)
    m = strategy_metrics(ret, actual, pred)
    blocks = wf.drop_duplicates("train_end")
    modal = blocks["spec"].mode().iloc[0]
    m.update({
        "asset": label, "target_kind": target_kind,
        "start": wf["month"].iloc[0], "end": wf["month"].iloc[-1],
        "blocks": len(blocks), "modal_spec": modal,
        "modal_share": float((blocks["spec"] == modal).mean()),
        "median_train_abs_r": float(blocks["train_r"].abs().median()),
    })
    return m


def fmt_num(x, digits=2):
    return "n.a." if x is None or pd.isna(x) else f"{x:.{digits}f}"


def fmt_pct(x, digits=1):
    return "n.a." if x is None or pd.isna(x) else f"{100*x:.{digits}f}%"


def table_html(df: pd.DataFrame, columns: list[tuple[str, str, callable]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for key, _, formatter in columns:
            val = row.get(key)
            text = formatter(val) if formatter else str(val)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def line_svg(series: dict[str, pd.Series], width=900, height=330) -> str:
    all_values = pd.concat(series.values()).replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        return ""
    vmin, vmax = min(float(all_values.min()), 0), max(float(all_values.max()), 0)
    if vmax == vmin:
        vmax += 1
    pad_l, pad_r, pad_t, pad_b = 54, 20, 18, 38
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    all_idx = sorted(set().union(*(s.dropna().index for s in series.values())))
    xmap = {x: pad_l + i / max(1, len(all_idx) - 1) * plot_w for i, x in enumerate(all_idx)}
    y = lambda v: pad_t + (vmax - v) / (vmax - vmin) * plot_h
    colors = ["#38bdf8", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#fb7185"]
    paths, legend = [], []
    for i, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{xmap[k]:.1f},{y(float(v)):.1f}" for k, v in s.dropna().items() if k in xmap)
        paths.append(f"<polyline points='{pts}' fill='none' stroke='{colors[i%len(colors)]}' stroke-width='2.2'/>")
        legend.append(f"<span><i style='background:{colors[i%len(colors)]}'></i>{html.escape(name)}</span>")
    grid = []
    for frac in (0, .25, .5, .75, 1):
        value = vmax - frac * (vmax - vmin)
        yy = pad_t + frac * plot_h
        grid.append(f"<line x1='{pad_l}' y1='{yy:.1f}' x2='{width-pad_r}' y2='{yy:.1f}' stroke='#273449'/><text x='{pad_l-8}' y='{yy+4:.1f}' text-anchor='end'>{value:.1f}×</text>")
    first, last = all_idx[0], all_idx[-1]
    return f"<div class='legend'>{''.join(legend)}</div><svg viewBox='0 0 {width} {height}' role='img'><g class='axis'>{''.join(grid)}<text x='{pad_l}' y='{height-10}'>{first}</text><text x='{width-pad_r}' y='{height-10}' text-anchor='end'>{last}</text></g>{''.join(paths)}</svg>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, equity_ret = load_inputs()
    oni2 = oni_candidates(base, 2)
    m_candidates = metal_candidates(base)
    descriptive_rows, wf_summaries, wf_files = [], [], {}

    # ONI -> metal benchmark return.
    for metal in METALS:
        y = base[f"r_{metal}"]
        best = descriptive_best(y, oni2)
        descriptive_rows.append({"relationship": "ONI → metal", "target": metal, **best})
        wf = walk_forward(y, oni2, min_train=120, test_months=12, cost_bps=10, legs=1)
        wf.to_csv(OUT / f"walkforward_oni_{metal.lower()}.csv", index=False)
        wf_files[f"ONI → {metal}"] = wf
        wf_summaries.append(summarize_wf(metal, wf, "ONI → metal benchmark"))

    spy = equity_ret["SPY"]
    # ONI -> infrastructure excess return and metal -> infrastructure excess return.
    for ticker in EQUITIES:
        y = (equity_ret[ticker] - spy).dropna()
        for relationship, candidates, slug in (
            ("ONI → infrastructure", oni2, "oni"),
            ("metal → infrastructure", m_candidates, "metal"),
        ):
            best = descriptive_best(y, candidates)
            descriptive_rows.append({"relationship": relationship, "target": ticker, **best})
            wf = walk_forward(y, candidates, min_train=60, test_months=12, cost_bps=10, legs=2)
            wf.to_csv(OUT / f"walkforward_{slug}_{ticker}.csv", index=False)
            wf_files[f"{relationship} {ticker}"] = wf
            wf_summaries.append(summarize_wf(ticker, wf, relationship))

    desc = pd.DataFrame(descriptive_rows)
    summary = pd.DataFrame(wf_summaries)
    desc.to_csv(OUT / "descriptive_correlations.csv", index=False)
    summary.to_csv(OUT / "walkforward_summary.csv", index=False)

    # Same-month metal/equity correlation matrix (descriptive, excess equity returns).
    matrix_rows = []
    for ticker in EQUITIES:
        y = equity_ret[ticker] - spy
        for metal in METALS:
            r, p, n = pearson(base[f"r_{metal}"], y)
            matrix_rows.append({"equity": ticker, "metal": metal, "r": r, "p": p, "n": n})
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(OUT / "metal_equity_same_month_correlations.csv", index=False)

    # Publication-delay sensitivity for benchmark metals.
    sensitivity = []
    for delay in (1, 2, 3):
        cand = oni_candidates(base, delay)
        for metal in METALS:
            wf = walk_forward(base[f"r_{metal}"], cand, min_train=120, test_months=12, cost_bps=10, legs=1)
            s = summarize_wf(metal, wf, f"delay {delay}")
            sensitivity.append({"delay_months": delay, **s})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "publication_delay_sensitivity.csv", index=False)

    # Test-interval sensitivity: same expanding training rule, different freeze length.
    interval_sensitivity = []
    for interval in (6, 12, 24):
        for metal in METALS:
            wf = walk_forward(base[f"r_{metal}"], oni2, min_train=120, test_months=interval, cost_bps=10, legs=1)
            s = summarize_wf(metal, wf, f"interval {interval}")
            interval_sensitivity.append({"test_interval_months": interval, **s})
    interval_sensitivity = pd.DataFrame(interval_sensitivity)
    interval_sensitivity.to_csv(OUT / "test_interval_sensitivity.csv", index=False)

    # Existing package audit summaries.
    rolling = pd.read_csv(INPUT / "rolling_60m_correlation_summary.csv")
    hac = pd.read_csv(INPUT / "HAC_lag_tests.csv")

    # Cumulative strategy wealth for the three metal benchmark proxies.
    wealth = {}
    for metal in METALS:
        wf = wf_files[f"ONI → {metal}"]
        idx = pd.PeriodIndex(wf["month"], freq="M")
        wealth[metal] = pd.Series((1 + wf["strategy_net"].to_numpy()).cumprod(), index=idx)

    metal_summary = summary[summary["target_kind"] == "ONI → metal benchmark"].copy()
    oni_eq_summary = summary[summary["target_kind"] == "ONI → infrastructure"].copy()
    metal_eq_summary = summary[summary["target_kind"] == "metal → infrastructure"].copy()

    outcome = "No robust tradable effect"
    positive = summary[(summary["n"] >= 24) & (summary["hac_p"] < .05) & (summary["sharpe"] > 0)]
    if len(positive) >= 3:
        outcome = "Some effects survive out of sample"

    def relationship_table(frame):
        return table_html(frame, [
            ("asset", "Target", str), ("start", "OOS start", str), ("end", "OOS end", str),
            ("n", "Months", lambda x: f"{int(x)}" if pd.notna(x) else "n.a."),
            ("ic", "OOS IC", lambda x: fmt_num(x, 2)),
            ("ann_return", "Net annual return", lambda x: fmt_pct(x, 1)),
            ("sharpe", "Sharpe", lambda x: fmt_num(x, 2)),
            ("max_drawdown", "Max drawdown", lambda x: fmt_pct(x, 1)),
            ("hac_p", "HAC p", lambda x: fmt_num(x, 3)),
            ("modal_spec", "Most selected specification", str),
            ("modal_share", "Selection share", lambda x: fmt_pct(x, 0)),
        ])

    desc_display = desc.copy()
    desc_display["target"] = desc_display["target"].map(lambda x: NAMES.get(x, x))
    matrix_pivot = matrix.pivot(index="equity", columns="metal", values="r").reset_index()
    matrix_pivot["equity"] = matrix_pivot["equity"].map(lambda x: NAMES.get(x, x))

    html_text = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ENSO, metals and cooling infrastructure: walk-forward test</title>
<style>
:root{{--bg:#08111f;--card:#0f1b2d;--card2:#142238;--text:#e8eef7;--muted:#9fb0c5;--line:#26364e;--blue:#38bdf8;--green:#34d399;--amber:#f59e0b;--red:#fb7185}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#0b1424 50%,#08111f);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 24px 72px}} h1{{font-size:44px;line-height:1.08;letter-spacing:-.035em;margin:12px 0 16px;max-width:980px}} h2{{font-size:25px;margin:48px 0 15px;letter-spacing:-.015em}} h3{{font-size:18px;margin:0 0 8px}} p{{max-width:920px}} a{{color:#7dd3fc}} .eyebrow{{color:var(--blue);text-transform:uppercase;letter-spacing:.14em;font-weight:700;font-size:12px}} .lede{{font-size:19px;color:#c7d3e3;max-width:940px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}} .card{{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:14px;padding:20px}} .metric{{font-size:29px;font-weight:760;letter-spacing:-.025em}} .label{{color:var(--muted);font-size:13px;margin-top:2px}} .good{{color:var(--green)}} .warn{{color:var(--amber)}} .bad{{color:var(--red)}} .callout{{border-left:4px solid var(--amber);padding:14px 18px;background:#191a22;border-radius:0 10px 10px 0;margin:18px 0}} .method{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .step{{border-top:2px solid var(--blue);background:var(--card);padding:14px;border-radius:8px}} .step b{{display:block;margin-bottom:4px}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:14px 0 24px}} table{{border-collapse:collapse;width:100%;min-width:780px;background:var(--card)}} th{{text-align:left;color:#bcd0e8;background:#14243a;font-size:12px;text-transform:uppercase;letter-spacing:.05em}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}} td{{font-variant-numeric:tabular-nums}} tr:last-child td{{border-bottom:0}} .small{{color:var(--muted);font-size:13px}} .chart{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:18px 0}} svg{{width:100%;height:auto}} .axis text{{fill:var(--muted);font-size:11px}} .legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:0 0 6px 50px}} .legend i{{width:18px;height:3px;display:inline-block;margin:0 6px 3px 0}} code{{background:#18263a;padding:2px 5px;border-radius:4px}} details{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 17px;margin:10px 0}} summary{{cursor:pointer;font-weight:700}} footer{{border-top:1px solid var(--line);margin-top:44px;padding-top:18px;color:var(--muted)}}
@media(max-width:800px){{h1{{font-size:34px}}.grid,.method{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class='eyebrow'>Research report · 12 September 2026</div>
<h1>ENSO, metals and cooling infrastructure: a leakage-controlled walk-forward test</h1>
<p class='lede'>The revised 1992–July 2026 ONI/benchmark-metal package contains weak and unstable lead–lag relationships. After imposing a two-month ONI publication delay, selecting the signal and lag only inside each training sample, freezing it for the next 12 months and charging turnover, the evidence does not support a robust trading rule.</p>
<div class='grid'>
<div class='card'><div class='metric bad'>{outcome}</div><div class='label'>Overall result after walk-forward evaluation</div></div>
<div class='card'><div class='metric'>2 months</div><div class='label'>Base-case ONI publication delay, plus one month to forecast the target return</div></div>
<div class='card'><div class='metric warn'>Futures not run</div><div class='label'>Massive API returned HTTP 403 for futures entitlement</div></div>
</div>

<h2>What was tested</h2>
<div class='method'>
<div class='step'><b>1. Information clock</b>At target month <i>t</i>, the newest usable centered ONI is <i>t−3</i>: two months of publication delay plus a one-month forecast horizon.</div>
<div class='step'><b>2. Training only</b>Each annual origin searches ONI and ΔONI with 0–12 additional lags. Metal-to-equity models search three metal returns with 1–6 lags.</div>
<div class='step'><b>3. Freeze</b>The highest absolute in-sample correlation and its OLS coefficients are held fixed for the next 12 months.</div>
<div class='step'><b>4. Trade and roll</b>Positions use only the frozen model and newly available lagged predictor. The model is re-estimated after the test block.</div>
</div>
<p class='small'>Metal models use a 120-month minimum training sample. Equity models use 60 months because VRT, TT and CARR have short post-listing histories. Metal strategies charge 10 bp per unit of turnover. Equity tests trade the stock’s return minus SPY and charge 10 bp on each of the two legs. September 2026 partial bars are excluded; all tests end in July 2026.</p>

<h2>ONI → metal benchmark returns</h2>
{relationship_table(metal_summary)}
<div class='chart'><h3>Net growth of $1 in the frozen-specification signal</h3><p class='small'>These are benchmark-price proxy strategies, not futures P&amp;L. A 1.0 ending value means no net gain.</p>{line_svg(wealth)}</div>
<p>The supplied full-sample lag sweep found several nominally significant extrema, but the 60-month rolling correlations move through zero and reach both positive and negative regimes. That pattern is consistent with specification instability, and the modal walk-forward specification shares above show that no single lag dominates training origins.</p>

<h2>ONI → cooling-infrastructure excess returns</h2>
{relationship_table(oni_eq_summary.assign(asset=oni_eq_summary['asset'].map(lambda x:NAMES.get(x,x))))}
<p>VRT, TT and CARR have limited post-2020 history, so their out-of-sample samples are especially small. Results for MOD, JCI and ETN have longer histories but still depend on a vendor history floor near 2003–2004. The equity target is monthly split-adjusted price return less SPY, not dividend total return.</p>

<h2>Metal returns → cooling-infrastructure excess returns</h2>
{relationship_table(metal_eq_summary.assign(asset=metal_eq_summary['asset'].map(lambda x:NAMES.get(x,x))))}
<p>This tests whether last month’s (or earlier) copper, aluminum or nickel benchmark return predicts the next equity excess return. It does not assert a cost pass-through mechanism; input contracts, hedging, geographic mix and data-center exposure differ by issuer.</p>

<h2>Descriptive correlations</h2>
<p>These full-sample values are diagnostic only. They are not model-selection evidence because the entire sample is visible.</p>
{table_html(desc_display, [('relationship','Relationship',str),('target','Target',str),('spec','Best full-sample specification',str),('n','N',lambda x:str(int(x))),('r','Pearson r',lambda x:fmt_num(x,3)),('p','Raw p',lambda x:fmt_num(x,3))])}
<h3>Same-month metal return / equity excess-return matrix</h3>
{table_html(matrix_pivot, [('equity','Equity',str),('Copper','Copper r',lambda x:fmt_num(x,2)),('Aluminum','Aluminum r',lambda x:fmt_num(x,2)),('Nickel','Nickel r',lambda x:fmt_num(x,2))])}

<h2>Publication-delay sensitivity</h2>
{table_html(sensitivity, [('delay_months','Delay',lambda x:f'{int(x)} month' if int(x)==1 else f'{int(x)} months'),('asset','Metal',str),('n','OOS months',lambda x:str(int(x))),('ic','OOS IC',lambda x:fmt_num(x,2)),('ann_return','Net annual return',lambda x:fmt_pct(x,1)),('sharpe','Sharpe',lambda x:fmt_num(x,2)),('hac_p','HAC p',lambda x:fmt_num(x,3))])}
<p>NOAA states that ONI is a centered three-month mean, updates the page by the fifth of each month and warns that values can change for up to two months after the initial real-time posting. The base case therefore delays the centered value by two months, then uses it to forecast the following month. This prevents timing leakage but does not recreate historical vintages; only an archived release series can do that.</p>

<h3>Frozen test-interval sensitivity</h3>
{table_html(interval_sensitivity, [('test_interval_months','Test interval',lambda x:f'{int(x)} months'),('asset','Metal',str),('n','OOS months',lambda x:str(int(x))),('ic','OOS IC',lambda x:fmt_num(x,2)),('ann_return','Net annual return',lambda x:fmt_pct(x,1)),('sharpe','Sharpe',lambda x:fmt_num(x,2)),('hac_p','HAC p',lambda x:fmt_num(x,3)),('modal_share','Modal-spec share',lambda x:fmt_pct(x,0))])}

<h2>Massive futures status</h2>
<div class='callout'><b>Contract-level futures backtest not executed.</b> The existing Massive key returned HTTP 403 (“not entitled to this data”) for <code>/futures/v1/products</code>. The same credential successfully returned stock aggregates. Massive documents CME, CBOT, COMEX and NYMEX futures coverage, with reference history beginning 3 April 2017 and full history restricted to higher plans. It does not claim LME coverage, so COMEX copper is the directly relevant supported contract; the supplied aluminum and nickel benchmark series cannot be relabeled as exchange futures.</div>
<p>A valid rerun must: enumerate expired contracts with a point-in-time date; retain first/last trade and settlement dates; choose the front/second/third expiry using only information at each decision date; compute each monthly return inside the same raw contract; prohibit returns that cross contracts; and use a predeclared roll rule based on then-known expiry or contemporaneous volume. Massive’s contract endpoint supports point-in-time lookups and its aggregate endpoint returns contract-specific OHLC, settlement price, volume and session date.</p>

<h2>Interpretation</h2>
<div class='grid'>
<div class='card'><h3>Correlation is regime-dependent</h3><p class='small'>The supplied rolling summaries span roughly −0.51 to +0.40 across metals/signals. A full-sample extreme is not a stable forecasting law.</p></div>
<div class='card'><h3>Selection consumes the apparent edge</h3><p class='small'>Each origin chooses among 26 ONI specifications or 18 metal specifications. Out-of-sample IC, net returns and HAC p-values are the relevant evidence.</p></div>
<div class='card'><h3>Cooling equities are not metal proxies</h3><p class='small'>They reflect broad equity beta, orders, margins and AI/data-center capital spending. Using SPY excess returns reduces, but does not remove, those confounds.</p></div>
</div>
<p><b>Decision:</b> treat the hypothesis as exploratory. Do not deploy a directional strategy from these results. The next defensible test is the 2017+ COMEX HG raw-contract walk-forward rerun after Massive futures entitlement is enabled, followed by a locked holdout that is not used for further lag or universe selection.</p>
<p class='small'>One sensitivity cell—copper with a three-month publication delay—has a nominal HAC p-value below 0.05. It is one of nine delay/metal combinations and was not predeclared as the base case, so it does not survive a simple Bonferroni threshold of 0.05/9.</p>

<h2>Audit trail and limitations</h2>
<details open><summary>Input package</summary><p>The benchmark file has {len(base):,} monthly rows from {base.index.min()} through {base.index.max()}. It contains revised NOAA ONI plus copper, aluminum and nickel benchmark prices. The supplied HAC table has {len(hac)} selected-lag tests, and the rolling summary has {len(rolling)} series summaries. Prices are benchmark levels, not individual futures contracts.</p></details>
<details><summary>Equity data</summary><p>Monthly OHLC aggregates came from Massive’s stock REST API with <code>adjusted=true</code>. Massive describes the adjustment as split adjustment. Dividends were not added. VRT begins February 2020 and CARR begins April 2020. TT is restricted to March 2020 onward to avoid mixing the pre-2008 ticker history with the present company.</p></details>
<details><summary>Statistical limits</summary><p>Raw p-values do not correct for the candidate search. HAC p-values test the mean strategy return with three Newey–West lags, not the full model-selection process. Annualized results from short samples are fragile. Transaction costs omit bid–ask spread, borrow, margin, roll slippage, contract multipliers and taxes.</p></details>
<details><summary>ONI vintages</summary><p>The two-month delay is a conservative timing proxy applied to today’s revised values. It prevents premature use of recent seasons but cannot reproduce past preliminary readings or later revisions. Archived release vintages remain required for a strict real-time simulation.</p></details>

<h2>Sources</h2>
<ul>
<li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/'>NOAA CPC ONI definition, update timing and revision warning</a></li>
<li><a href='https://massive.com/docs/rest/futures/overview'>Massive Futures REST overview and venue coverage</a></li>
<li><a href='https://massive.com/docs/rest/futures/contracts'>Massive futures contract reference endpoint</a></li>
<li><a href='https://massive.com/docs/rest/futures/aggregates'>Massive contract aggregate bars endpoint and history/plan table</a></li>
<li><a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>Massive stock aggregate bars endpoint</a></li>
<li><a href='https://www.vertiv.com/4a47fc/globalassets/documents/white-papers/vertiv-variable-cooling-technolog-data-center-wp-en-na-sl-70653_312508_0.pdf'>Vertiv thermal-management reference</a></li>
<li><a href='https://www.modine.com/wp-content/uploads/2025/07/Modine-2025-SustainabilityReport-FINAL.pdf'>Modine data-center cooling reference</a></li>
<li><a href='https://www.tranetechnologies.com/en/index/company/data-centers.html'>Trane Technologies data-center cooling portfolio</a></li>
</ul>
<footer>Prepared from the user-supplied ENSO/metal package and Massive API responses. Research use only; no investment recommendation.</footer>
</main></body></html>"""
    (ROOT / "ENSO_metals_cooling_walkforward_report.html").write_text(html_text)

    provenance_files = [
        INPUT / "aligned_monthly_ONI_metals_1992_2026-07.csv",
        INPUT / "HAC_lag_tests.csv",
        INPUT / "rolling_60m_correlation_summary.csv",
        *(RAW / f"massive_{ticker}_monthly.json" for ticker in EQUITIES + ["SPY"]),
        RAW / "massive_futures_entitlement_check.json",
    ]
    provenance = {
        "created_at": "2026-09-12",
        "massive_stock_endpoint_pattern": "/v2/aggs/ticker/{ticker}/range/1/month/1980-01-01/2026-09-11?adjusted=true&sort=asc&limit=50000",
        "massive_futures_check_endpoint": "/futures/v1/products?product_code=HG&limit=100",
        "api_credentials_in_output": False,
        "files": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in provenance_files
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    results = {
        "as_of": "2026-09-12",
        "outcome": outcome,
        "oni_publication_delay_months": 2,
        "test_interval_months": 12,
        "futures_status": "not_run_http_403_entitlement",
        "walkforward_summary": summary.replace({np.nan: None}).to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"report": str(ROOT / "ENSO_metals_cooling_walkforward_report.html"), "outcome": outcome, "rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
