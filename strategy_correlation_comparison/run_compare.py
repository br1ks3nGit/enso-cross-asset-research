#!/usr/bin/env python3
"""Compare cached raw correlations without re-estimating prior strategies."""

from pathlib import Path
import html
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ENERGY = Path("enso_energy_peaks/calculated")
PRIOR = Path("enso_metals_walkforward/calculated")


def fisher_ci(r, n):
    if n <= 3 or abs(r) >= 1:
        return math.nan, math.nan
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def main():
    ROOT.mkdir(exist_ok=True)
    energy = pd.read_csv(ENERGY / "predictive_correlations.csv")
    prior = pd.read_csv(PRIOR / "descriptive_correlations.csv")
    robust = pd.read_csv(ENERGY / "surviving_correlation_robustness.csv")

    # Conservative energy screen: globally BH-significant rows, one strongest row per company.
    screened = energy[(energy["ticker"].isin(["COP", "PSX", "MPC", "SLB"])) & (energy["q_bh_global"] < .05)].copy()
    selected_energy = screened.loc[screened.groupby("ticker")["r"].apply(lambda s: s.abs().idxmax())].copy()
    selected_energy["family"] = "ONI → energy"
    selected_energy["relationship"] = selected_energy.apply(lambda r: f"{r['signal']} → {r['ticker']} minus {r['benchmark']} ({int(r['horizon_months'])}m)", axis=1)
    selected_energy["p_method"] = "HAC"
    selected_energy["q"] = selected_energy["q_bh_global"]

    # Prior families are read from their saved full-sample table; no re-estimation.
    prior_sig = prior[prior["p"] < .05].copy()
    leaders = []
    for relationship, family in [
        ("ONI → metal", "ONI → metals"),
        ("ONI → infrastructure", "ONI → cooling infrastructure"),
        ("metal → infrastructure", "Metals → cooling infrastructure"),
    ]:
        g = prior_sig[prior_sig["relationship"] == relationship]
        row = g.loc[g["r"].abs().idxmax()].copy()
        leaders.append({
            "family": family, "relationship": f"{row['spec']} → {row['target']} (1m target)",
            "n": row["n"], "r": row["r"], "p": row["p"], "p_method": "Pearson raw", "q": math.nan,
            "ticker": row["target"], "benchmark": "n.a.", "signal": row["spec"], "horizon_months": 1,
        })
    prior_leaders = pd.DataFrame(leaders)

    columns = ["family", "relationship", "n", "r", "p", "p_method", "q", "ticker", "benchmark", "signal", "horizon_months"]
    candidates = pd.concat([selected_energy[columns], prior_leaders[columns]], ignore_index=True)
    candidates["abs_r"] = candidates["r"].abs()
    candidates["r_squared"] = candidates["r"] ** 2
    cis = candidates.apply(lambda r: fisher_ci(r["r"], int(r["n"])), axis=1)
    candidates["ci_low"] = [x[0] for x in cis]
    candidates["ci_high"] = [x[1] for x in cis]
    candidates = candidates.sort_values("abs_r", ascending=False).reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates.to_csv(ROOT / "raw_correlation_candidates.csv", index=False)

    # Same one-month horizon comparison.
    energy_1m = energy[(energy["ticker"].isin(["COP", "PSX", "MPC", "SLB"])) & (energy["horizon_months"] == 1)].copy()
    energy_1m = energy_1m.loc[[energy_1m["r"].abs().idxmax()]]
    monthly = pd.concat([
        pd.DataFrame([{
            "family": "ONI → energy", "relationship": f"{energy_1m.iloc[0]['signal']} → {energy_1m.iloc[0]['ticker']} minus {energy_1m.iloc[0]['benchmark']}",
            "n": energy_1m.iloc[0]["n"], "r": energy_1m.iloc[0]["r"], "p": energy_1m.iloc[0]["p"],
        }]),
        prior_leaders[["family", "relationship", "n", "r", "p"]],
    ], ignore_index=True)
    monthly["abs_r"] = monthly["r"].abs()
    monthly["r_squared"] = monthly["r"] ** 2
    monthly = monthly.sort_values("abs_r", ascending=False).reset_index(drop=True)
    monthly.to_csv(ROOT / "one_month_horizon_comparison.csv", index=False)

    energy_native = candidates[candidates["family"] == "ONI → energy"].copy()
    omitted = energy[~energy["ticker"].isin(["COP", "PSX", "MPC", "SLB", "XLE"])].copy()
    best_omitted = omitted.loc[omitted.groupby("ticker")["p"].idxmin()].sort_values("p")
    best_omitted.to_csv(ROOT / "excluded_energy_companies.csv", index=False)

    def pct(x): return "n.a." if pd.isna(x) else f"{100*x:.1f}%"
    def num(x, d=3): return "n.a." if pd.isna(x) else f"{x:.{d}f}"
    def tab(df, cols):
        h = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
        rows = []
        for _, row in df.iterrows():
            rows.append("<tr>" + "".join(f"<td>{html.escape(fn(row[k]) if fn else str(row[k]))}</td>" for k, _, fn in cols) + "</tr>")
        return f"<div class='table'><table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"

    best = candidates.iloc[0]
    best_month = monthly.iloc[0]
    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Raw correlation strategy comparison</title><style>
:root{{--bg:#08121e;--card:#111e2e;--text:#edf3f9;--muted:#9eafc2;--line:#293a50;--blue:#38bdf8;--green:#34d399;--amber:#fbbf24}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07101c,#0c1827);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:44px 24px 70px}}h1{{font-size:42px;line-height:1.08;letter-spacing:-.03em;max-width:900px}}h2{{font-size:24px;margin-top:42px}}p{{max-width:900px}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:750;letter-spacing:.14em;text-transform:uppercase}}.lede{{font-size:19px;color:#c8d5e4}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:19px}}.metric{{font-size:28px;font-weight:760}}.label,.small{{color:var(--muted);font-size:13px}}.callout{{border-left:4px solid var(--amber);padding:14px 18px;background:#1d1e21;border-radius:0 9px 9px 0}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:14px 0 24px}}table{{border-collapse:collapse;width:100%;min-width:800px;background:var(--card)}}th{{text-align:left;background:#16283d;color:#c2d3e5;font-size:12px;text-transform:uppercase}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}footer{{border-top:1px solid var(--line);margin-top:42px;padding-top:16px;color:var(--muted)}}@media(max-width:760px){{.cards{{grid-template-columns:1fr}}h1{{font-size:33px}}}}
</style></head><body><main><div class='eyebrow'>Cached-results comparison · no prior strategy re-estimation</div><h1>Which relationship is the best candidate for a stable beta?</h1><p class='lede'>Raw correlation ranks ONI → energy first when each strategy keeps its original horizon. On a common one-month horizon, metals → cooling infrastructure ranks first. Correlation can identify a candidate relationship, but it cannot identify the largest beta without the predictor and target volatilities.</p>
<div class='cards'><div class='card'><div class='metric'>{best['r']:.3f}</div><div class='label'>Strongest native-horizon r: {html.escape(best['relationship'])}</div></div><div class='card'><div class='metric'>{pct(best['r_squared'])}</div><div class='label'>Raw variance association, r²—not causal explanatory power</div></div><div class='card'><div class='metric'>{best_month['r']:.3f}</div><div class='label'>Strongest common one-month r: {html.escape(best_month['relationship'])}</div></div></div>
<div class='callout'><b>Selection rule.</b> “Leave highest p-value companies” is interpreted as remove them. Energy candidates are restricted to companies with at least one globally BH-significant ONI relationship: COP, PSX, MPC and SLB. XOM, CVX, OXY, EOG, KMI and WMB are excluded from the candidate ranking.</div>
<h2>Native-horizon ranking</h2><p>Each row preserves the horizon used in its saved analysis. The energy rows use overlapping forward returns and HAC p-values. Prior-strategy rows use their saved one-month Pearson correlations.</p>
{tab(candidates,[('rank','Rank',lambda x:str(int(x))),('family','Strategy family',str),('relationship','Raw relationship',str),('n','N',lambda x:str(int(x))),('r','r',lambda x:num(x,3)),('r_squared','r²',lambda x:pct(x)),('ci_low','95% CI low',lambda x:num(x,3)),('ci_high','95% CI high',lambda x:num(x,3)),('p','p-value',lambda x:num(x,4)),('p_method','p method',str),('q','Global q',lambda x:num(x,4))])}
<h2>Common one-month target</h2><p>This is the fairer raw-correlation comparison because longer cumulative horizons mechanically smooth returns and overlap observations.</p>
{tab(monthly,[('family','Strategy family',str),('relationship','Relationship',str),('n','N',lambda x:str(int(x))),('r','r',lambda x:num(x,3)),('r_squared','r²',lambda x:pct(x)),('p','p-value',lambda x:num(x,4))])}
<h2>Energy correlation stability</h2><p>The energy winners were already split into pre-2015, post-2015 and COVID-excluded samples. These are reused results, not new model searches.</p>
{tab(robust,[('ticker','Ticker',str),('benchmark','Benchmark',str),('signal','Signal',str),('horizon_months','Horizon',lambda x:f'{int(x)}m'),('sample','Sample',str),('n','N',lambda x:str(int(x))),('r','r',lambda x:num(x,3)),('p','HAC p',lambda x:num(x,4))])}
<h2>Decision</h2><p><b>Best native-horizon beta candidate:</b> ONI → COP relative to XLE over six months, because it has the largest raw magnitude (r={best['r']:.3f}), the tightest association among the compared families (r²={pct(best['r_squared'])}) and survives the global energy screen. Its relationship is negative.</p><p><b>Best common-horizon candidate:</b> {html.escape(best_month['relationship'])}, r={best_month['r']:.3f}. This prevents the six- and twelve-month energy targets from receiving an unfair smoothing advantage.</p><p><b>Recommended beta research order:</b> first estimate and validate COP/XLE on a locked future holdout; second test nickel/VRT on a longer history; third test the one-month PSX/XLE ΔONI relationship. Raw r ranks linear fit, not beta magnitude, economic causality or expected P&amp;L.</p>
<footer>Built only from existing saved correlation outputs. No Massive download or previous strategy re-estimation was performed.</footer></main></body></html>"""
    (ROOT / "raw_correlation_strategy_comparison.html").write_text(report)
    print(candidates[["rank", "family", "relationship", "n", "r", "p", "q"]].to_string(index=False))


if __name__ == "__main__":
    main()
