#!/usr/bin/env python3
"""Build an evidence-led ENSO/data-center cooling capex and supplier report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CALC = ROOT / "calculated"
FORECAST = ROOT.parent / "enso_forecast_2026" / "calculated" / "state_enso_effects.json"
TEMPERATURES = ROOT.parent / "enso_forecast_2026" / "calculated" / "state_temperature.jsonl.gz"
PRIOR_SPEND = ROOT.parent / "enso_datacenter_cooling_2026" / "calculated" / "affected_states_cooling_spend.csv"

# Separately disclosed power capacity in Lau & Tsai (2026), Table 1.
CAPACITY_2025_GW = {
    "Illinois": 1.746,
    "Indiana": 0.061,
    "Iowa": 0.660,
    "Michigan": 0.127,
    "North Dakota": 1.090,
    "Ohio": 0.892,
}
CAPACITY_2030_GW = {
    "Illinois": 7.079,
    "Indiana": 4.861,
    "Iowa": 2.600,
    "Michigan": 1.531,
    "North Dakota": 2.422,
    "Ohio": 9.726,
}

# Cost inputs are a transparent screen, not an invoice forecast.
# JLL: Chicago shell/core $12m-$14m per MW. AAON: $7b-$9b North American
# cooling equipment and infrastructure TAM against ~$41b construction put in place.
BUILD_COST_M_PER_MW = {"low": 12.0, "central": 13.0, "high": 14.0}
COOLING_SHARE = {"low": 7.0 / 41.0, "central": 8.0 / 41.0, "high": 9.0 / 41.0}

# Hypothetical audit finding: share of fleet touched and share of a complete cooling
# system replaced/augmented. The central demand forecast remains zero because the
# weather scenario does not breach a design-capacity screen and arrives too late.
INCIDENCE = {"low": 0.005, "central": 0.010, "high": 0.020}
SCOPE = {"low": 0.05, "central": 0.05, "high": 0.10}

OPERATORS = {
    "North Dakota": "No named operator validated in the cited project sources",
    "Minnesota": "Meta Rosemount; Google Hermantown and Pine Island in development",
    "Wisconsin": "Microsoft Mount Pleasant campus",
    "South Dakota": "No named operator validated in the cited project sources",
    "Iowa": "Meta Altoona; Google Council Bluffs; Google Cedar Rapids in development",
    "Michigan": "Google Van Buren Township in development",
    "Illinois": "Meta DeKalb; broad colocation market",
    "Indiana": "Meta Jeffersonville and Lebanon; Google Fort Wayne and three projects in development",
    "Ohio": "Meta New Albany and Bowling Green; Google Columbus, Lancaster and New Albany",
    "Montana": "No named operator validated in the cited project sources",
    "New York": "Large operating market; no project-level vendor match validated",
}

SUPPLIERS = [
    {
        "ticker": "MOD", "company": "Modine / Airedale", "role": "Chillers, fan walls, CRAHs, controls",
        "purity": "Very high", "evidence": "$348.6m FY27 Q1 Data Centers sales; $4bn 2027-29 customer capacity agreement",
        "state_angle": "Headquartered in Wisconsin; national delivery", "rank": 1,
    },
    {
        "ticker": "AAON", "company": "AAON / BASX", "role": "CDUs, liquid cooling, zero-water free-cooling chillers",
        "purity": "Very high", "evidence": "$345m Q2 BASX-branded sales; $2.0bn total backlog; named customer withheld",
        "state_angle": "National delivery; factories outside selected states", "rank": 2,
    },
    {
        "ticker": "VRT", "company": "Vertiv", "role": "End-to-end thermal, liquid cooling, heat rejection and service",
        "purity": "High but also power", "evidence": "$13.5-$14.0bn FY26 total-sales guidance; $15.0bn backlog at 2025 year-end",
        "state_angle": "Headquartered in Ohio; national delivery and service", "rank": 3,
    },
    {
        "ticker": "NVT", "company": "nVent", "role": "Liquid-cooling connections, enclosures and distribution",
        "purity": "Medium-high", "evidence": "Third liquid-cooling capacity expansion in three years; 400,000+ sq ft added",
        "state_angle": "New 160,000 sq ft liquid-cooling site in Blaine, Minnesota", "rank": 4,
    },
    {
        "ticker": "JCI", "company": "Johnson Controls / YORK / Silent-Aire", "role": "Chillers, controls and full thermal chain",
        "purity": "Medium", "evidence": "$6.1bn Q2 sales; $20bn backlog; data centers cited as a growth driver",
        "state_angle": "Major Wisconsin operating base; national delivery", "rank": 5,
    },
    {
        "ticker": "TT", "company": "Trane Technologies", "role": "Large chillers, controls and services",
        "purity": "Medium-low", "evidence": "$6.4bn Q2 sales; $12.1bn backlog; Americas Commercial HVAC bookings +50%",
        "state_angle": "National delivery; no cited state-project supplier match", "rank": 6,
    },
    {
        "ticker": "SPXC", "company": "SPX Technologies", "role": "Heat-rejection and cooling-tower adjacency",
        "purity": "Low / unverified", "evidence": "Public thermal-equipment exposure; no selected-state hyperscaler award validated",
        "state_angle": "Do not treat as a pure data-center cooling proxy", "rank": 7,
    },
    {
        "ticker": "FIX", "company": "Comfort Systems USA", "role": "Mechanical/electrical installation and modular construction",
        "purity": "Second-order", "evidence": "$3.27bn Q2 revenue; $14.06bn backlog; data centers are a disclosed demand driver",
        "state_angle": "Installer, not cooling-equipment OEM", "rank": 8,
    },
    {
        "ticker": "EME", "company": "EMCOR", "role": "Mechanical/electrical construction and facility services",
        "purity": "Second-order", "evidence": "$5.15bn Q2 revenue; $17.14bn remaining performance obligations",
        "state_angle": "Installer, not cooling-equipment OEM", "rank": 9,
    },
]


def load_states() -> pd.DataFrame:
    estimates = pd.DataFrame(json.loads(FORECAST.read_text())["estimates"])
    states = estimates[estimates["enso_effect_c"] > 2.0].copy()
    temps = pd.read_json(TEMPERATURES, lines=True)
    normals = (
        temps[(temps["month"] == 12) & temps["year"].between(1991, 2020)]
        .groupby("state", as_index=False)["temperature_c"].mean()
        .rename(columns={"temperature_c": "dec_normal_c"})
    )
    prior = pd.read_csv(PRIOR_SPEND)[[
        "state", "central_incremental_spend_usd_per_100mw",
        "central_incremental_spend_usd_disclosed_capacity",
    ]]
    states = states.merge(normals, on="state", how="left").merge(prior, on="state", how="left")
    states["implied_dec_mean_c"] = states["dec_normal_c"] + states["enso_effect_c"]
    states["implied_dec_p90_mean_c"] = states["dec_normal_c"] + states["p90_c"]
    states["capacity_2025_gw"] = states["state"].map(CAPACITY_2025_GW)
    states["capacity_2030_gw"] = states["state"].map(CAPACITY_2030_GW)
    states["operator_examples"] = states["state"].map(OPERATORS)

    for scenario in ("low", "central", "high"):
        equipment_m_per_mw = BUILD_COST_M_PER_MW[scenario] * COOLING_SHARE[scenario]
        states[f"equipment_basis_m_per_mw_{scenario}"] = equipment_m_per_mw
        states[f"audit_screen_usd_per_100mw_{scenario}"] = (
            100 * equipment_m_per_mw * 1_000_000 * INCIDENCE[scenario] * SCOPE[scenario]
        )
        states[f"audit_screen_usd_{scenario}"] = (
            states["capacity_2025_gw"] * 1000 * equipment_m_per_mw * 1_000_000
            * INCIDENCE[scenario] * SCOPE[scenario]
        )
    states["weather_driven_capex_central_usd"] = 0.0
    states["pipeline_increment_gw"] = states["capacity_2030_gw"] - states["capacity_2025_gw"]
    states["pipeline_cooling_basis_usd"] = (
        states["pipeline_increment_gw"] * 1000
        * BUILD_COST_M_PER_MW["central"] * COOLING_SHARE["central"] * 1_000_000
    )
    return states.sort_values("enso_effect_c", ascending=False).reset_index(drop=True)


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    series: dict[str, pd.Series] = {}
    for path in sorted((ROOT / "raw").glob("massive_*_daily.json")):
        ticker = path.name.split("_")[1]
        frame = pd.DataFrame(json.loads(path.read_text())["results"])
        frame["date"] = pd.to_datetime(frame["t"], unit="ms").dt.normalize()
        series[ticker] = frame.set_index("date")["c"].rename(ticker)
    prices = pd.DataFrame(series).sort_index()
    rows = []
    periods = {
        "ytd_2026": "2026-01-02",
        "apr_to_signal": "2026-04-01",
        "jul_to_signal": "2026-07-01",
    }
    for ticker in prices.columns:
        available = prices[ticker].dropna()
        ytd = available.loc["2026-01-02":"2026-09-11"]
        record = {"ticker": ticker, "last_date": str(available.index[-1].date()), "last_close": float(available.iloc[-1])}
        for name, start in periods.items():
            window = available.loc[start:"2026-09-11"]
            record[name] = float(window.iloc[-1] / window.iloc[0] - 1)
        record["drawdown_from_2026_high"] = float(ytd.iloc[-1] / ytd.max() - 1)
        rows.append(record)
    stats = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    return prices, stats


def table(frame: pd.DataFrame, columns: list[tuple[str, str, object]], min_width: int = 800) -> str:
    heads = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row.get(key)
            rendered = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table' style='--min:{min_width}px'><table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def line_chart(prices: pd.DataFrame) -> str:
    tickers = ["VRT", "MOD", "AAON", "JCI", "TT", "NVT", "SPXC"]
    px = prices.loc["2026-01-02":"2026-09-11", tickers + ["XLI", "XLK"]].dropna()
    normalized = px / px.iloc[0] * 100
    chart = pd.DataFrame({
        "Cooling supplier basket": normalized[tickers].mean(axis=1),
        "XLI": normalized["XLI"],
        "XLK": normalized["XLK"],
    })
    width, height, left, right, top, bottom = 930, 380, 58, 24, 30, 42
    ymin, ymax = float(chart.min().min()), float(chart.max().max())
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad
    colors = {"Cooling supplier basket": "#0f5b78", "XLI": "#c7602b", "XLK": "#6a6f78"}
    parts = [f"<rect width='{width}' height='{height}' fill='#ffffff'/>"]
    for value in range(int(ymin // 10 * 10), int(ymax // 10 * 10 + 11), 10):
        y = top + (ymax - value) / (ymax - ymin) * (height - top - bottom)
        parts.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#d9dde2'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#59616a'>{value}</text>")
    for name in chart.columns:
        values = chart[name]
        coords = []
        for i, value in enumerate(values):
            x = left + i / (len(values) - 1) * (width - left - right)
            y = top + (ymax - value) / (ymax - ymin) * (height - top - bottom)
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(f"<polyline points='{' '.join(coords)}' fill='none' stroke='{colors[name]}' stroke-width='2.4'/>")
    legend_x = left
    for name in chart.columns:
        parts.append(f"<line x1='{legend_x}' x2='{legend_x+24}' y1='16' y2='16' stroke='{colors[name]}' stroke-width='3'/>")
        parts.append(f"<text x='{legend_x+30}' y='20' font-size='12' fill='#20262c'>{html.escape(name)}</text>")
        legend_x += 225
    parts.append(f"<text x='{width/2}' y='{height-10}' text-anchor='middle' font-size='12' fill='#59616a'>2 Jan 2026 to 11 Sep 2026 · 100 = first close</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Massive adjusted price index for cooling supplier basket, XLI and XLK'>{''.join(parts)}</svg>"


def make_report(states: pd.DataFrame, prices: pd.DataFrame, price_stats: pd.DataFrame) -> str:
    fmt_usd = lambda x: "n.a." if pd.isna(x) else f"${float(x):,.0f}"
    fmt_usdm = lambda x: "n.a." if pd.isna(x) else f"${float(x)/1_000_000:,.2f}m"
    fmt_gw = lambda x: "not separately disclosed" if pd.isna(x) else f"{float(x):.3f}"
    fmt_c = lambda x: "n.a." if pd.isna(x) else f"{float(x):+.2f}"
    fmt_pct = lambda x: "n.a." if pd.isna(x) else f"{100*float(x):+.1f}%"
    fmt_price = lambda x: "n.a." if pd.isna(x) else f"${float(x):,.2f}"

    disclosed = states[states["capacity_2025_gw"].notna()].copy()
    disclosed_gw = disclosed["capacity_2025_gw"].sum()
    stress_low = disclosed["audit_screen_usd_low"].sum()
    stress_central = disclosed["audit_screen_usd_central"].sum()
    stress_high = disclosed["audit_screen_usd_high"].sum()
    pipeline_gw = disclosed["pipeline_increment_gw"].sum()
    pipeline_cooling = disclosed["pipeline_cooling_basis_usd"].sum()

    state_table = table(states, [
        ("state", "State", None),
        ("enso_effect_c", "ENSO ΔT, °C", fmt_c),
        ("dec_normal_c", "1991–2020 Dec mean, °C", fmt_c),
        ("implied_dec_mean_c", "Normal + ENSO, °C", fmt_c),
        ("implied_dec_p90_mean_c", "Normal + p90 effect, °C", fmt_c),
        ("capacity_2025_gw", "2025 disclosed GW", fmt_gw),
        ("weather_driven_capex_central_usd", "Central ENSO capex", fmt_usd),
        ("audit_screen_usd_central", "Contingency screen", fmt_usdm),
        ("central_incremental_spend_usd_disclosed_capacity", "Prior Dec electricity opex", fmt_usdm),
    ], 1120)

    project_table = table(states, [
        ("state", "State", None),
        ("operator_examples", "Current / developing operator examples", None),
    ], 850)

    supplier_frame = pd.DataFrame(SUPPLIERS)
    supplier_table = table(supplier_frame, [
        ("rank", "Rank", lambda x: str(int(x))),
        ("ticker", "Ticker", None),
        ("company", "Public company", None),
        ("role", "Cooling-chain role", None),
        ("purity", "Exposure", None),
        ("evidence", "Current demand evidence", None),
        ("state_angle", "Selected-state relevance", None),
    ], 1260)

    merged_market = supplier_frame[["ticker", "company", "rank"]].merge(price_stats, on="ticker", how="left").sort_values("rank")
    market_table = table(merged_market, [
        ("ticker", "Ticker", None),
        ("company", "Company", None),
        ("last_close", "11 Sep close", fmt_price),
        ("ytd_2026", "2026 YTD", fmt_pct),
        ("apr_to_signal", "1 Apr–11 Sep", fmt_pct),
        ("jul_to_signal", "1 Jul–11 Sep", fmt_pct),
        ("drawdown_from_2026_high", "From 2026 high", fmt_pct),
    ], 930)

    direct = price_stats[price_stats["ticker"].isin(["VRT", "MOD", "AAON", "JCI", "TT", "NVT", "SPXC"])]
    basket_ytd = direct["ytd_2026"].mean()
    basket_apr = direct["apr_to_signal"].mean()
    basket_jul = direct["jul_to_signal"].mean()
    xli = price_stats.set_index("ticker").loc["XLI"]

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ENSO 2026 · data-center cooling capex and public suppliers</title><style>
:root{{--ink:#15202a;--muted:#5f6b75;--paper:#f5f2ea;--panel:#ffffff;--line:#cfd4d8;--navy:#0f3b52;--blue:#0f5b78;--orange:#c7602b;--red:#9c2d2d;--green:#286848}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.58 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1220px;margin:auto;padding:42px 24px 72px}}h1{{font:700 42px/1.08 Georgia,serif;letter-spacing:-.02em;max-width:1050px;margin:12px 0 16px}}h2{{font:700 27px/1.2 Georgia,serif;margin:46px 0 12px;color:var(--navy)}}h3{{font-size:18px;margin:29px 0 8px}}p,li{{max-width:1000px}}a{{color:var(--blue);text-underline-offset:2px}}.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--orange);font-weight:750}}.lede{{font-size:19px;color:#39444e;max-width:1040px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:26px 0}}.metric{{background:var(--panel);padding:18px}}.metric b{{display:block;font:700 28px/1.15 Georgia,serif;color:var(--navy)}}.metric span{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}.verdict{{border-left:5px solid var(--red);background:#fff;padding:17px 20px;margin:23px 0}}.note{{border-left:4px solid var(--orange);background:#fff;padding:14px 18px;margin:18px 0}}.positive{{border-left-color:var(--green)}}.table{{overflow:auto;border:1px solid var(--line);background:var(--panel);margin:14px 0 24px}}table{{border-collapse:collapse;width:100%;min-width:var(--min)}}th,td{{padding:10px 11px;border-bottom:1px solid #dde1e4;text-align:right;vertical-align:top}}th{{font-size:11px;letter-spacing:.035em;text-transform:uppercase;background:#edf0f2;color:#49545d;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}td:nth-child(2){{text-align:left}}code{{background:#e6e9eb;padding:2px 5px}}svg{{width:100%;height:auto;border:1px solid var(--line);background:white;margin:15px 0 20px}}.small{{font-size:13px;color:var(--muted)}}.label{{display:inline-block;padding:2px 6px;border:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}footer{{border-top:1px solid var(--line);margin-top:45px;padding-top:18px;color:var(--muted);font-size:13px}}@media(max-width:850px){{.metrics,.split{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}}}@media(max-width:560px){{.metrics,.split{{grid-template-columns:1fr}}}}
</style></head><body><main><div class='eyebrow'>Deep-research update · information available through 13 September 2026</div><h1>The December ENSO signal raises cooling electricity use, but it does not create a credible 2026 chiller-order wave</h1><p class='lede'>For the {disclosed_gw:.3f} GW of affected-state capacity separately disclosed in the source data, the central estimate of <strong>new cooling equipment capex attributable only to this ENSO forecast is $0</strong>. A deliberately conservative post-audit contingency screen is about <strong>${stress_central/1_000_000:.1f} million</strong>, with a ${stress_low/1_000_000:.1f}–${stress_high/1_000_000:.1f} million scenario range. This is tiny beside the roughly ${pipeline_cooling/1_000_000_000:.0f} billion cooling-equipment basis associated with the already-announced 2025–30 capacity pipeline in the same six states.</p>
<div class='metrics'><div class='metric'><span>Central ENSO equipment capex</span><b>$0</b></div><div class='metric'><span>Contingency audit screen</span><b>${stress_central/1_000_000:.1f}m</b></div><div class='metric'><span>Disclosed pipeline growth</span><b>{pipeline_gw:.1f} GW</b></div><div class='metric'><span>Average equipment lead time</span><b>33 weeks</b></div></div>
<div class='verdict'><strong>Investment conclusion.</strong> The structural AI/data-center cooling theme is real; the September 2026 ENSO forecast is not a material incremental revenue catalyst for public cooling suppliers. Orders are already being placed for multi-year capacity buildouts, the relevant December sites remain cold, and the signal-to-season window is only about 12–16 weeks versus a reported 33-week average equipment lead time. Treat VRT, MOD, AAON, NVT, JCI or TT as structural infrastructure exposures—not as a clean one-season ENSO trade.</div>

<h2>What would appear on the balance sheet?</h2><p><strong>New chillers, CDUs, pumps, fan walls and heat-rejection equipment</strong> are normally capitalized by a data-center owner as construction in progress and then transferred to property, plant and equipment when placed in service. Cash paid is capital expenditure in investing cash flow; depreciation reaches the income statement later. The <strong>${disclosed['central_incremental_spend_usd_disclosed_capacity'].sum()/1_000_000:.2f} million</strong> central December electricity estimate from the earlier report is different: it is operating expense, not equipment capex.</p>
<p>The $0 central capex figure means no <em>additional identifiable PP&amp;E caused by this weather forecast</em>. It does not mean operators are spending nothing on cooling. The six states' published capacity pipeline rises from {disclosed_gw:.3f} GW in 2025 to {disclosed['capacity_2030_gw'].sum():.3f} GW in 2030; using the cost screen below, that expansion embeds about ${pipeline_cooling/1_000_000_000:.1f} billion of cooling equipment and infrastructure. That is planned growth capex and must not be relabeled “El Niño capex.” Capacity figures come from <a href='https://doi.org/10.1021/acs.energyfuels.6c01309'>Lau &amp; Tsai (2026), Table 1</a>; the paper itself notes that disclosed/announced capacity is incomplete.</p>

<h2>Why the modeled +2°C to +4°C does not imply more installed capacity</h2><p>The prior forecast estimates an ENSO-associated contribution, not a forecast of the entire temperature departure. Joining it to <a href='https://www.ncei.noaa.gov/pub/data/cirs/climdiv/'>NOAA state temperature data</a> shows that the implied 1991–2020-normal-plus-ENSO December state means range from about −5.1°C in North Dakota to +2.7°C in Ohio. Even at the forecast effect's p90, every selected-state monthly mean remains below +4.4°C. Those temperatures reduce economizer/free-cooling hours and raise auxiliary electricity; they do not test the summer outdoor-design condition that determines installed heat-rejection capacity.</p>
<p><a href='https://www.ashrae.org/technical-resources/ai-data-center-framework/energy-and-thermal-efficiency'>ASHRAE's AI data-center framework</a> recommends climate-zone economization, modular liquid-cooling systems and N+1 or 2N pumping/heat-exchange redundancy. That engineering architecture is why a warm winter month normally consumes more energy before it triggers emergency equipment procurement. The original state effects and limitations remain in the <a href='../enso_forecast_2026/ENSO_forecast_September_December_2026.html'>September–December ENSO forecast</a>.</p>
{state_table}
<p class='small'>“Normal + p90 effect” adds the upper 80% interval endpoint for the ENSO contribution to the 1991–2020 December mean. It still excludes unrelated weather variability and is not an extreme hourly dry-bulb or wet-bulb design temperature.</p>

<h2>How the contingency screen is calculated</h2><p>This screen answers a narrower question: if a post-season engineering audit nevertheless finds small deficiencies, how much balance-sheet capex might be touched?</p>
<p><code>Cooling replacement basis per MW = build cost per MW × cooling share</code></p><p><code>Audit contingency capex = affected MW × replacement basis × incidence × scope</code></p>
<ul><li>Build cost: $12m / $13m / $14m per MW. <a href='https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook.pdf'>JLL's 2026 report</a> gives $12m–$14m/MW for a Chicago 50 MW air-cooled shell-and-core facility and notes a 10% liquid-cooling premium.</li><li>Cooling share: 17.1% / 19.5% / 22.0%, calculated from <a href='https://www.aaon.com/hubfs/Documents/IR%20Documents/IR%20Presentations/2026/Investor%20Presentation_June%202%202026.pdf'>AAON's June 2026 investor presentation</a>: $7bn–$9bn North American cooling equipment/infrastructure TAM divided by about $41bn of data-center construction put in place.</li><li>Audit incidence: 0.5% / 1% / 2% of affected capacity; scoped addition: 5% / 5% / 10% of a complete cooling system. These are explicit analyst assumptions, not measured fleet failure rates.</li></ul>
<p>The resulting central replacement basis is ${(BUILD_COST_M_PER_MW['central']*COOLING_SHARE['central']):.2f} million per MW. The contingency is ${states['audit_screen_usd_per_100mw_central'].iloc[0]/1000:,.0f} thousand per 100 MW, before applying a known site-specific deficiency. Since no breach is identified, the contingency is <strong>not</strong> the central forecast and should not be booked as committed capex.</p>

<h2>Current builders and operators in the affected states</h2><p>These examples establish that there is real construction activity, not that a particular OEM won a particular state contract. <a href='https://datacenters.atmeta.com/us-locations/'>Meta lists facilities</a> in Illinois, Indiana, Iowa, Minnesota and Ohio. <a href='https://www.datacenters.google/'>Google lists operating or developing sites</a> in Indiana, Iowa, Michigan, Minnesota and Ohio. <a href='https://local.microsoft.com/blog/mount-pleasant-datacenter-project-update/'>Microsoft's Mount Pleasant, Wisconsin project</a> runs building development and equipment testing through fall 2026.</p>{project_table}
<div class='note'><strong>Vendor-matching limitation.</strong> Hyperscalers often withhold supplier and site details. The research found a confirmed <a href='https://investors.modine.com/news/news-details/2024/Corscale-Data-Centers-approves-IST-of-Airedale-by-Modine-cooling-solutions-on-72MW-Gainesville-Crossing-Data-Center/default.aspx'>Airedale/Modine deployment for Corscale's 72 MW Virginia project</a> and an <a href='https://www.aaon.com/news/aaon-awarded-liquid-cooling-data-center-orders-totaling-approximately-174.5-million'>AAON/BASX $174.5m order from one unnamed data-center customer</a>, but no source ties those awards to the selected states. I therefore do not assign invented project revenue by state.</div>

<h2>Public suppliers that can benefit from the structural cooling buildout</h2><p>All tickers below returned adjusted U.S. daily bars from Massive through 11 September 2026. The ranking is based on directness of cooling exposure and current demand evidence—not valuation, expected return or a site-level contract claim.</p>{supplier_table}
<p><strong>Best operating exposure:</strong> MOD and AAON are the cleanest cooling-equipment names; VRT is the strongest broad critical-infrastructure platform; NVT is a focused liquid-cooling enabler with a new Minnesota plant; JCI and TT are diversified HVAC suppliers. FIX and EME are installation beneficiaries when projects proceed, but they are not cooling-equipment OEMs.</p>
<p>The demand is already visible in company disclosures. <a href='https://investors.modine.com/news/news-details/2026/Modine-Reports-First-Quarter-Fiscal-2027-Results/default.aspx'>Modine reported $348.6m of quarterly Data Centers sales</a> and separately announced a <a href='https://investors.modine.com/news/news-details/2026/Modine-Announces-Landmark-4-Billion-Long-Term-Capacity-Agreement-through-2029-with-Strategic-Data-Center-Customer-for-Airedale-by-Modine-Cooling-Solutions/default.aspx'>$4bn customer capacity agreement for 2027–29</a>. <a href='https://investors.aaon.com/investor-news/aaon-reports-record-second-quarter-2026-results-driven-by-strong-demand-accelerating-throughput-and-improved-operating-execution'>AAON reported $345m of Q2 BASX-branded sales and a $2bn total backlog</a>. <a href='https://investors.vertiv.com/news/news-details/2026/Vertiv-Reports-Strong-Fourth-Quarter-with-Organic-Orders-Growth-of-252-and-Diluted-EPS-Growth-of-200-Adjusted-Diluted-EPS-37/'>Vertiv entered 2026 with $15bn backlog</a>. <a href='https://investors.nvent.com/press-releases/press-release-details/2026/nVent-Expands-Data-Center-Liquid-Cooling-Capacity/default.aspx'>nVent's July 2026 expansion</a> was its third liquid-cooling capacity addition in three years.</p>

<h2>Revenue alteration caused by this ENSO forecast</h2><div class='split'><div><h3>Central case</h3><p><strong>$0 incremental supplier revenue.</strong> With no design-capacity breach and insufficient procurement time, weather-attributable equipment capex is $0, so <code>supplier revenue uplift = $0 × win share = $0</code>. Companies still grow from AI/data-center construction, but that is the baseline structural theme.</p></div><div><h3>Contingency case</h3><p>If the full ${stress_central/1_000_000:.1f}m audit screen were actually ordered, a supplier capturing 10%, 25% or 50% would receive only ${stress_central*0.10/1_000_000:.2f}m, ${stress_central*0.25/1_000_000:.2f}m or ${stress_central*0.50/1_000_000:.2f}m. Even the entire screen is just {100*stress_central/4_000_000_000:.3f}% of Modine's $4bn agreement, {100*stress_central/2_000_000_000:.3f}% of AAON's total backlog and {100*stress_central/15_000_000_000:.3f}% of Vertiv's year-end backlog.</p></div></div>

<h2>Have operators already adapted?</h2><p><strong>Substantially, yes—because of AI density, water and efficiency constraints, not this ENSO forecast.</strong> <a href='https://www.microsoft.com/en-us/microsoft-cloud/blog/2024/12/09/sustainable-by-design-next-generation-datacenters-consume-zero-water-for-cooling/'>Microsoft says all new designs beginning in August 2024 adopted closed-loop, zero-water-for-cooling technology</a>. <a href='https://about.fb.com/news/2026/08/closed-loop-cooling-explained-the-plumbing-behind-metas-ai/'>Meta says most of its newest AI-optimized centers use closed-loop liquid cooling</a> and describes air-assisted liquid cooling for older facilities. <a href='https://www.datacenters.google/efficiency/'>Google reports trailing-twelve-month PUE of 1.05–1.11</a> at its cited Ohio and Iowa campuses, which indicates highly optimized existing cooling operations.</p>
<p>Timing is also adverse for a September weather trade. <a href='https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook-new.pdf'>JLL reports a 33-week average data-center equipment lead time</a>, while the forecast-to-December window is only about 12–16 weeks. Large equipment serving December 2026 was therefore almost certainly specified or ordered before this signal.</p>

<h2>Market test: was the cooling theme already in prices?</h2><p>The equal-weight seven-OEM basket returned {basket_ytd:+.1%} from the first 2026 close to 11 September, versus {xli['ytd_2026']:+.1%} for XLI. But from 1 July to the signal date the basket lost {abs(basket_jul):.1%}, worse than XLI's {abs(xli['jul_to_signal']):.1%} decline. This is consistent with a heavily anticipated, volatile AI-infrastructure theme; it is not evidence of an ENSO premium. Adjusted closes are from Massive. Returns exclude dividends, tax and transaction costs and are descriptive, not a causal event study.</p>{line_chart(prices)}{market_table}
<p>The basket's 1 April–11 September return was {basket_apr:+.1%}. Dispersion is large: VRT, MOD and NVT had strong 2026 YTD gains, while several names were far below their 2026 highs by the signal date. That makes “already priced” nuanced: the structural growth narrative was public and in backlogs, but individual equities had partially corrected. It does <strong>not</strong> rescue an ENSO-specific strategy because the modeled revenue increment is immaterial.</p>

<h2>Decision rule</h2><ul><li><strong>Do not trade this as September-forecast → December cooling-equipment revenue.</strong> The central incremental capex/revenue mechanism is absent.</li><li><strong>Monitor operational data instead:</strong> local hourly dry-bulb/wet-bulb temperatures, economizer hours, PUE, cooling alarms, rental cooling, and emergency purchase orders. State monthly mean ΔT is too coarse.</li><li><strong>For structural exposure, separate the thesis:</strong> new AI capacity, liquid-cooling adoption, order growth, backlog conversion, manufacturing yield and valuation—not ENSO.</li><li><strong>Revisit only if</strong> a facility-level forecast crosses design conditions, a repeat multi-season warming pattern changes engineering standards, or operators disclose weather-driven retrofit orders.</li></ul>

<h2>Sources</h2><ul><li><a href='../enso_forecast_2026/ENSO_forecast_September_December_2026.html'>Existing ENSO forecast and state estimates</a>; <a href='https://www.ncei.noaa.gov/pub/data/cirs/climdiv/'>NOAA state temperature archive</a>; <a href='https://doi.org/10.1021/acs.energyfuels.6c01309'>Lau &amp; Tsai state capacity table</a>.</li><li><a href='https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook.pdf'>JLL 2026 construction costs</a>; <a href='https://www.aaon.com/hubfs/Documents/IR%20Documents/IR%20Presentations/2026/Investor%20Presentation_June%202%202026.pdf'>AAON 2026 cooling TAM</a>; <a href='https://www.ashrae.org/technical-resources/ai-data-center-framework/energy-and-thermal-efficiency'>ASHRAE thermal-efficiency framework</a>.</li><li><a href='https://datacenters.atmeta.com/us-locations/'>Meta U.S. locations</a>; <a href='https://www.datacenters.google/'>Google locations</a>; <a href='https://local.microsoft.com/blog/mount-pleasant-datacenter-project-update/'>Microsoft Wisconsin project</a>.</li><li><a href='https://investors.modine.com/news/news-details/2026/Modine-Reports-First-Quarter-Fiscal-2027-Results/default.aspx'>Modine Q1 FY27</a>; <a href='https://investors.aaon.com/investor-news/aaon-reports-record-second-quarter-2026-results-driven-by-strong-demand-accelerating-throughput-and-improved-operating-execution'>AAON Q2 2026</a>; <a href='https://investors.vertiv.com/news/news-details/2026/Vertiv-Reports-Strong-Fourth-Quarter-with-Organic-Orders-Growth-of-252-and-Diluted-EPS-Growth-of-200-Adjusted-Diluted-EPS-37/'>Vertiv Q4 2025</a>; <a href='https://investors.nvent.com/press-releases/press-release-details/2026/nVent-Expands-Data-Center-Liquid-Cooling-Capacity/default.aspx'>nVent capacity expansion</a>.</li><li><a href='https://investors.johnsoncontrols.com/news/news-details/2026/Johnson-Controls-Reports-Strong-Q2-Results-Raises-FY26-Guidance/default.aspx'>Johnson Controls Q2 2026</a>; <a href='https://ir.tranetechnologies.com/news-and-events/news-releases/news-release-details/2026/Trane-Technologies-Reports-Strong-Second-Quarter-Results-Raises-Full-Year-Revenue-and-EPS-Guidance/default.aspx'>Trane Q2 2026</a>; <a href='https://investors.comfortsystemsusa.com/node/18636'>Comfort Systems Q2 2026</a>; <a href='https://emcorgroup.com/investor-relations/press-releases/2026-news/emcor-group-inc-reports-second-quarter-2026-results'>EMCOR Q2 2026</a>.</li><li><a href='https://www.microsoft.com/en-us/microsoft-cloud/blog/2024/12/09/sustainable-by-design-next-generation-datacenters-consume-zero-water-for-cooling/'>Microsoft cooling design</a>; <a href='https://about.fb.com/news/2026/08/closed-loop-cooling-explained-the-plumbing-behind-metas-ai/'>Meta cooling design</a>; <a href='https://www.datacenters.google/efficiency/'>Google PUE disclosure</a>; <a href='https://www.jll.com/content/dam/jllcom/en/global/documents/reports/research-reports/26-research-global-data-center-outlook-new.pdf'>JLL lead-time data</a>.</li></ul>
<footer>Prepared 13 September 2026. Dollar values are USD. This is a scenario analysis, not investment advice, a company revenue forecast, or an engineering design study. Massive prices are adjusted daily closes through 11 September 2026.</footer></main></body></html>"""


def main() -> None:
    CALC.mkdir(parents=True, exist_ok=True)
    states = load_states()
    prices, price_stats = load_prices()
    state_columns = [
        "state", "enso_effect_c", "p10_c", "p90_c", "dec_normal_c",
        "implied_dec_mean_c", "implied_dec_p90_mean_c", "capacity_2025_gw",
        "capacity_2030_gw", "weather_driven_capex_central_usd",
        "audit_screen_usd_low", "audit_screen_usd_central", "audit_screen_usd_high",
        "audit_screen_usd_per_100mw_low", "audit_screen_usd_per_100mw_central",
        "audit_screen_usd_per_100mw_high", "pipeline_increment_gw",
        "pipeline_cooling_basis_usd", "central_incremental_spend_usd_per_100mw",
        "central_incremental_spend_usd_disclosed_capacity", "operator_examples",
    ]
    states[state_columns].to_csv(CALC / "state_cooling_capex_screen.csv", index=False)
    price_stats.to_csv(CALC / "massive_price_diagnostics.csv", index=False)
    pd.DataFrame(SUPPLIERS).to_csv(CALC / "public_supplier_screen.csv", index=False)
    pd.DataFrame([
        {"scenario": k, "build_cost_m_per_mw": BUILD_COST_M_PER_MW[k],
         "cooling_share": COOLING_SHARE[k], "equipment_basis_m_per_mw": BUILD_COST_M_PER_MW[k] * COOLING_SHARE[k],
         "fleet_incidence": INCIDENCE[k], "scope_of_full_system": SCOPE[k]}
        for k in ("low", "central", "high")
    ]).to_csv(CALC / "capex_assumptions.csv", index=False)
    (ROOT / "ENSO_2026_cooling_capex_supplier_research.html").write_text(make_report(states, prices, price_stats))
    disclosed = states[states["capacity_2025_gw"].notna()]
    print(f"states={len(states)} disclosed_gw={disclosed.capacity_2025_gw.sum():.3f}")
    print(f"central_weather_capex=${disclosed.weather_driven_capex_central_usd.sum():,.0f}")
    print(f"audit_screen=${disclosed.audit_screen_usd_central.sum():,.0f}")
    print(f"pipeline_increment={disclosed.pipeline_increment_gw.sum():.3f} GW")


if __name__ == "__main__":
    main()
