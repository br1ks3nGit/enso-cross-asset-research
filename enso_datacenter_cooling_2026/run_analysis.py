#!/usr/bin/env python3
"""Estimate December 2026 data-center cooling spend for states with ENSO ΔT > 2°C."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calculated"
FORECAST = ROOT.parent / "enso_forecast_2026" / "calculated" / "state_enso_effects.json"
SOURCE_REPORT = ROOT.parent / "enso_forecast_2026" / "ENSO_forecast_September_December_2026.html"

# June 2026 average commercial retail electricity price, EIA Table 5.6.A, cents/kWh.
COMMERCIAL_RATE = {
    "Illinois": 14.53,
    "Indiana": 14.15,
    "Iowa": 12.79,
    "Michigan": 16.63,
    "Minnesota": 13.68,
    "Montana": 13.48,
    "North Dakota": 8.02,
    "New York": 23.56,
    "Ohio": 13.77,
    "South Dakota": 11.35,
    "Wisconsin": 14.05,
}

# Separately disclosed 2025 state capacity from Lau & Tsai (2026), Table 1.
# This is total data-center power capacity, not IT load. The other selected states
# are included in the paper's aggregate "Others" row and are deliberately left blank.
DISCLOSED_CAPACITY_GW_2025 = {
    "Illinois": 1.746,
    "Indiana": 0.061,
    "Iowa": 0.660,
    "Michigan": 0.127,
    "North Dakota": 1.090,
    "Ohio": 0.892,
}

HOURS_DECEMBER = 31 * 24
FACILITY_CAPACITY_UTILIZATION = 0.70
BASE_PUE = 1.30
COOLING_PUE_COMPONENT = 0.24  # 80% of the 0.30 non-IT PUE increment; scenario assumption.
KAPPA = {"low": 0.0075, "central": 0.0150, "high": 0.0300}  # ΔPUE per °C.


def load_affected_states() -> pd.DataFrame:
    payload = json.loads(FORECAST.read_text())
    frame = pd.DataFrame(payload["estimates"])
    affected = frame[frame["enso_effect_c"] > 2.0].copy()
    if affected.empty:
        raise RuntimeError("No state-month estimates exceed 2°C")
    affected["month_name"] = affected["month"].map({9: "September", 10: "October", 11: "November", 12: "December"})
    affected["commercial_rate_cents_kwh"] = affected["state"].map(COMMERCIAL_RATE)
    if affected["commercial_rate_cents_kwh"].isna().any():
        missing = affected.loc[affected["commercial_rate_cents_kwh"].isna(), "state"].tolist()
        raise RuntimeError(f"Missing EIA rate for {missing}")
    affected["capacity_gw_2025_disclosed"] = affected["state"].map(DISCLOSED_CAPACITY_GW_2025)
    affected["capacity_status"] = affected["capacity_gw_2025_disclosed"].notna().map({True: "Separately disclosed", False: "Included in paper's aggregate Others row"})
    return affected.sort_values("enso_effect_c", ascending=False).reset_index(drop=True)


def calculate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    facility_mw = 100.0
    average_it_mw = facility_mw * FACILITY_CAPACITY_UTILIZATION / BASE_PUE
    out["average_it_mw_per_100mw_facility"] = average_it_mw
    out["central_delta_pue"] = KAPPA["central"] * out["enso_effect_c"]
    out["central_incremental_cooling_mwh_per_100mw"] = average_it_mw * HOURS_DECEMBER * out["central_delta_pue"]
    out["central_incremental_spend_usd_per_100mw"] = (
        out["central_incremental_cooling_mwh_per_100mw"]
        * out["commercial_rate_cents_kwh"] * 10.0
    )
    out["low_incremental_spend_usd_per_100mw"] = (
        average_it_mw * HOURS_DECEMBER * KAPPA["low"] * out["p10_c"].clip(lower=0)
        * out["commercial_rate_cents_kwh"] * 10.0
    )
    out["high_incremental_spend_usd_per_100mw"] = (
        average_it_mw * HOURS_DECEMBER * KAPPA["high"] * out["p90_c"].clip(lower=0)
        * out["commercial_rate_cents_kwh"] * 10.0
    )
    out["central_cooling_spend_increase_pct"] = out["central_delta_pue"] / COOLING_PUE_COMPONENT
    out["low_cooling_spend_increase_pct"] = KAPPA["low"] * out["p10_c"].clip(lower=0) / COOLING_PUE_COMPONENT
    out["high_cooling_spend_increase_pct"] = KAPPA["high"] * out["p90_c"].clip(lower=0) / COOLING_PUE_COMPONENT
    out["central_incremental_energy_mwh_disclosed_capacity"] = (
        out["central_incremental_cooling_mwh_per_100mw"]
        * out["capacity_gw_2025_disclosed"] * 10.0
    )
    out["central_incremental_spend_usd_disclosed_capacity"] = (
        out["central_incremental_spend_usd_per_100mw"]
        * out["capacity_gw_2025_disclosed"] * 10.0
    )
    out["low_incremental_spend_usd_disclosed_capacity"] = (
        out["low_incremental_spend_usd_per_100mw"]
        * out["capacity_gw_2025_disclosed"] * 10.0
    )
    out["high_incremental_spend_usd_disclosed_capacity"] = (
        out["high_incremental_spend_usd_per_100mw"]
        * out["capacity_gw_2025_disclosed"] * 10.0
    )
    return out


def table(frame: pd.DataFrame, columns: list[tuple[str, str, object]]) -> str:
    heads = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row.get(key)
            text = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table><thead><tr>{heads}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def bar_chart(frame: pd.DataFrame) -> str:
    data = frame.sort_values("central_incremental_spend_usd_per_100mw", ascending=True)
    width, height, left, right, top, bottom = 980, 490, 145, 50, 28, 35
    max_value = float(data["high_incremental_spend_usd_per_100mw"].max())
    usable = width - left - right
    row_h = (height - top - bottom) / len(data)
    parts = []
    for i, (_, row) in enumerate(data.iterrows()):
        y = top + i * row_h + 4
        low = float(row["low_incremental_spend_usd_per_100mw"]) / max_value * usable
        central = float(row["central_incremental_spend_usd_per_100mw"]) / max_value * usable
        high = float(row["high_incremental_spend_usd_per_100mw"]) / max_value * usable
        parts.append(f"<text x='{left-10}' y='{y+18:.1f}' text-anchor='end' fill='#d7e3ee' font-size='13'>{html.escape(row['state'])}</text>")
        parts.append(f"<line x1='{left+low:.1f}' x2='{left+high:.1f}' y1='{y+10:.1f}' y2='{y+10:.1f}' stroke='#64748b' stroke-width='5'/>")
        parts.append(f"<circle cx='{left+central:.1f}' cy='{y+10:.1f}' r='6' fill='#38bdf8'/>")
        parts.append(f"<text x='{left+central+10:.1f}' y='{y+15:.1f}' fill='white' font-size='12'>${central/usable*max_value/1000:.0f}k</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Incremental December cooling spend per 100 MW'><rect width='100%' height='100%' fill='#0f1d2d'/>{''.join(parts)}</svg>"


def make_report(frame: pd.DataFrame) -> str:
    usd = lambda x: "n.a." if pd.isna(x) else f"${float(x):,.0f}"
    usd_m = lambda x: "n.a." if pd.isna(x) else f"${float(x)/1_000_000:,.2f}m"
    pct = lambda x: "n.a." if pd.isna(x) else f"{100*float(x):.1f}%"
    num = lambda x: "n.a." if pd.isna(x) else f"{float(x):,.1f}"
    deg = lambda x: "n.a." if pd.isna(x) else f"{float(x):+.2f}"
    gw = lambda x: "not separately disclosed" if pd.isna(x) else f"{float(x):.3f}"

    disclosed = frame[frame["capacity_gw_2025_disclosed"].notna()].copy()
    missing = frame[frame["capacity_gw_2025_disclosed"].isna()].copy()
    central_total = disclosed["central_incremental_spend_usd_disclosed_capacity"].sum()
    low_total = disclosed["low_incremental_spend_usd_disclosed_capacity"].sum()
    high_total = disclosed["high_incremental_spend_usd_disclosed_capacity"].sum()
    energy_total = disclosed["central_incremental_energy_mwh_disclosed_capacity"].sum()
    disclosed_gw = disclosed["capacity_gw_2025_disclosed"].sum()

    normalized_table = table(frame, [
        ("state", "State", None),
        ("enso_effect_c", "December ΔT, °C", deg),
        ("p10_c", "80% interval low", deg),
        ("p90_c", "80% interval high", deg),
        ("commercial_rate_cents_kwh", "Electricity, ¢/kWh", num),
        ("central_incremental_cooling_mwh_per_100mw", "Extra cooling MWh / 100 MW", num),
        ("central_incremental_spend_usd_per_100mw", "Central extra spend / 100 MW", usd),
        ("low_incremental_spend_usd_per_100mw", "Low", usd),
        ("high_incremental_spend_usd_per_100mw", "High", usd),
        ("central_cooling_spend_increase_pct", "Modeled cooling-spend increase", pct),
    ])
    absolute_table = table(disclosed.sort_values("central_incremental_spend_usd_disclosed_capacity", ascending=False), [
        ("state", "State", None),
        ("capacity_gw_2025_disclosed", "Disclosed 2025 capacity, GW", gw),
        ("central_incremental_energy_mwh_disclosed_capacity", "Extra cooling MWh", num),
        ("central_incremental_spend_usd_disclosed_capacity", "Central extra December spend", usd_m),
        ("low_incremental_spend_usd_disclosed_capacity", "Low", usd_m),
        ("high_incremental_spend_usd_disclosed_capacity", "High", usd_m),
    ])
    missing_table = table(missing, [
        ("state", "State", None),
        ("enso_effect_c", "December ΔT, °C", deg),
        ("central_incremental_spend_usd_per_100mw", "Extra spend per 100 MW", usd),
        ("capacity_status", "Why no state total", None),
    ])
    chart = bar_chart(frame)

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ENSO temperature and data-center cooling spend</title><style>
:root{{--card:#101e2e;--text:#eef5fb;--muted:#a6b5c7;--line:#2b3d52;--blue:#38bdf8;--amber:#fbbf24;--green:#34d399}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07101b,#0b1928);color:var(--text);font:15px/1.58 Inter,system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:46px 24px 76px}}h1{{font-size:42px;line-height:1.08;letter-spacing:-.03em;max-width:1000px}}h2{{font-size:26px;margin-top:44px}}h3{{font-size:19px;margin-top:30px}}p,li{{max-width:980px}}a{{color:#7dd3fc}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:760;letter-spacing:.14em;text-transform:uppercase}}.lede{{font-size:19px;color:#cad7e5}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px}}.metric{{font-size:27px;font-weight:760}}.label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}.callout{{border-left:4px solid var(--amber);background:#172131;padding:15px 18px;border-radius:8px;margin:22px 0}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:15px 0 25px}}table{{border-collapse:collapse;width:100%;min-width:850px;background:#0c1927}}th,td{{padding:10px 12px;border-bottom:1px solid #22364a;text-align:right;white-space:nowrap}}th{{color:#b9c9d8;background:#112337;font-size:12px}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;margin:10px 0 20px}}code{{background:#12263a;padding:2px 5px;border-radius:4px}}.muted{{color:var(--muted)}}footer{{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}}}
</style></head><body><main><div class='eyebrow'>December 2026 conditional scenario · states above +2°C only</div><h1>ENSO-related warming could add about ${central_total/1_000_000:.1f} million to one month of cooling electricity</h1><p class='lede'>The estimate covers {disclosed_gw:.3f} GW of separately disclosed 2025 data-center power capacity across six affected states. It is a partial total: five other states exceed +2°C but lack separately reported MW in the capacity source.</p>
<div class='cards'><div class='card'><div class='label'>Affected states</div><div class='metric'>{len(frame)}</div></div><div class='card'><div class='label'>Central disclosed-state spend</div><div class='metric'>${central_total/1_000_000:.1f}m</div></div><div class='card'><div class='label'>Engineering + climate range</div><div class='metric'>${low_total/1_000_000:.1f}–${high_total/1_000_000:.1f}m</div></div><div class='card'><div class='label'>Extra electricity</div><div class='metric'>{energy_total/1000:.1f} GWh</div></div></div>
<div class='callout'><strong>Precise meaning:</strong> this is incremental December electricity spending attributable to the modeled ENSO temperature contribution, not total cooling opex, equipment capex, water cost or an annual run rate. September–November contain no state forecast above +2°C.</div>
<h2>All affected states, normalized per 100 MW</h2><p>This is the most comparable view because published state capacity is incomplete. The central scenario implies roughly 13%–25% more modeled cooling electricity during December, depending on state ΔT. It does not mean total data-center electricity rises by that amount; IT load remains unchanged.</p>{chart}{normalized_table}
<h2>Absolute estimate where capacity is disclosed</h2>{absolute_table}<p>The six-state central estimate is <strong>${central_total/1_000_000:.2f} million</strong>, with a combined climate-and-engineering scenario range of <strong>${low_total/1_000_000:.2f}–${high_total/1_000_000:.2f} million</strong>. Illinois dominates because its disclosed capacity is largest. North Dakota is second despite cheap electricity because its forecast ΔT is +4.0°C and disclosed capacity is 1.09 GW.</p>
<h3>States without a defensible absolute total</h3>{missing_table}<p>Multiplying these rows by guessed facility MW would create false precision. To estimate a specific operator or campus, multiply the per-100-MW result by its total facility power capacity divided by 100 MW.</p>
<h2>Calculation</h2><p>The model converts total facility capacity to average IT load, adds a temperature-driven PUE increment, and prices the extra energy:</p><p><code>Average IT MW = facility capacity MW × 70% utilization ÷ 1.30 PUE</code></p><p><code>ΔPUE = κ × ΔT</code>, where central <code>κ = 0.015 PUE/°C</code>, low = 0.0075 and high = 0.030.</p><p><code>Extra cooling MWh = average IT MW × 744 hours × ΔPUE</code></p><p><code>Extra spending = extra cooling MWh × EIA commercial electricity price</code></p><p><code>Cooling-spend increase = ΔPUE ÷ 0.24 modeled cooling-PUE component</code></p>
<h2>What is measured and what is assumed</h2><ul><li><strong>Measured/model-derived:</strong> state ENSO temperature contribution and its 80% interval from the existing forecast; June 2026 state commercial electricity rates from EIA; separately disclosed 2025 state data-center power capacity from Lau and Tsai.</li><li><strong>Scenario assumptions:</strong> 70% facility-capacity utilization, PUE 1.30, cooling equal to 0.24 PUE points, and κ from 0.0075 to 0.030 PUE/°C. κ is not an empirically estimated coefficient for each state's fleet.</li><li><strong>Capacity caveat:</strong> the paper reports announced/disclosed power capacity and excludes facilities without disclosed requirements. Its 2025 figure is therefore an incomplete proxy for December 2026 operating capacity.</li><li><strong>Temperature caveat:</strong> the state ΔT is an ENSO-associated contribution relative to the fitted ONI-neutral counterfactual. It is not the entire departure from climatology and does not resolve hourly wet-bulb temperature, humidity or economizer thresholds.</li><li><strong>Price caveat:</strong> hyperscale PPAs and tariffs can differ materially from EIA's average commercial retail price. Demand charges and hedges are excluded.</li></ul>
<h2>Operational interpretation</h2><ul><li><strong>Highest normalized dollar sensitivity:</strong> New York, Minnesota and Michigan because the combination of ΔT and electricity price is largest.</li><li><strong>Highest disclosed absolute exposure:</strong> Illinois and North Dakota, followed by Ohio and Iowa.</li><li><strong>Cooling vendors:</strong> the weather shock raises utilization and electricity consumption of installed cooling systems. A single warm month does not automatically create equivalent cooling-equipment revenue; capex responds mainly if repeated heat, capacity constraints or reliability requirements change design decisions.</li><li><strong>Modern liquid-cooled sites:</strong> higher coolant temperatures and economizers can reduce κ, placing them nearer the low scenario. Older air-cooled or chiller-heavy sites can sit nearer the high scenario.</li></ul>
<h2>Sources</h2><ul><li><a href='../enso_forecast_2026/ENSO_forecast_September_December_2026.html'>Existing September–December 2026 ENSO temperature forecast</a></li><li><a href='https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a'>EIA Electric Power Monthly, Table 5.6.A, June 2026 commercial prices</a></li><li><a href='https://doi.org/10.1021/acs.energyfuels.6c01309'>Lau &amp; Tsai (2026), state data-center power capacity</a></li><li><a href='https://eta-publications.lbl.gov/sites/default/files/2024-12/us_data_center_energy_usage_report_lbnl-2001637_0.pdf'>Lawrence Berkeley National Laboratory, 2024 U.S. Data Center Energy Usage Report</a></li></ul>
<footer>Prepared 13 September 2026. All dollar values are USD. Central and low/high results are scenario estimates, not invoice forecasts.</footer></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = calculate(load_affected_states())
    cols = [
        "state", "month_name", "enso_effect_c", "p10_c", "p90_c", "residual_weather_sd_c",
        "commercial_rate_cents_kwh", "capacity_gw_2025_disclosed", "capacity_status",
        "central_incremental_cooling_mwh_per_100mw", "central_incremental_spend_usd_per_100mw",
        "low_incremental_spend_usd_per_100mw", "high_incremental_spend_usd_per_100mw",
        "central_cooling_spend_increase_pct", "low_cooling_spend_increase_pct", "high_cooling_spend_increase_pct",
        "central_incremental_energy_mwh_disclosed_capacity", "central_incremental_spend_usd_disclosed_capacity",
        "low_incremental_spend_usd_disclosed_capacity", "high_incremental_spend_usd_disclosed_capacity",
    ]
    results[cols].to_csv(OUT / "affected_states_cooling_spend.csv", index=False)
    pd.DataFrame([
        {"assumption": "December hours", "value": HOURS_DECEMBER, "unit": "hours"},
        {"assumption": "Facility capacity utilization", "value": FACILITY_CAPACITY_UTILIZATION, "unit": "fraction"},
        {"assumption": "Base PUE", "value": BASE_PUE, "unit": "PUE"},
        {"assumption": "Cooling PUE component", "value": COOLING_PUE_COMPONENT, "unit": "PUE"},
        {"assumption": "Low temperature sensitivity", "value": KAPPA["low"], "unit": "ΔPUE per °C"},
        {"assumption": "Central temperature sensitivity", "value": KAPPA["central"], "unit": "ΔPUE per °C"},
        {"assumption": "High temperature sensitivity", "value": KAPPA["high"], "unit": "ΔPUE per °C"},
    ]).to_csv(OUT / "assumptions.csv", index=False)
    hashes = {
        "state_enso_effects.json": hashlib.sha256(FORECAST.read_bytes()).hexdigest(),
        "source_enso_report.html": hashlib.sha256(SOURCE_REPORT.read_bytes()).hexdigest(),
        "lau_tsai_2026_datacenter_capacity.pdf": hashlib.sha256((ROOT / "raw" / "lau_tsai_2026_datacenter_capacity.pdf").read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps({
        "scope": "States with central ENSO effect > 2°C; all occur in December 2026",
        "electricity_price_source": "EIA Electric Power Monthly Table 5.6.A, June 2026 commercial price",
        "capacity_source": "Lau and Tsai, Energy & Fuels 2026, Table 1, 2025 disclosed capacity",
        "capacity_policy": "No absolute state estimate where Table 1 groups the state into Others",
        "assumptions": {
            "hours": HOURS_DECEMBER, "utilization": FACILITY_CAPACITY_UTILIZATION,
            "base_pue": BASE_PUE, "cooling_pue_component": COOLING_PUE_COMPONENT,
            "delta_pue_per_c": KAPPA,
        },
        "hashes": hashes,
    }, indent=2) + "\n")
    report = make_report(results)
    (ROOT / "ENSO_2026_datacenter_cooling_spend.html").write_text(report)
    print(results[["state", "enso_effect_c", "central_incremental_spend_usd_per_100mw", "capacity_gw_2025_disclosed", "central_incremental_spend_usd_disclosed_capacity"]].to_string(index=False))


if __name__ == "__main__":
    main()
