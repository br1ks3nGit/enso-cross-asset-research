#!/usr/bin/env python3
"""Backtest five option implementations of the frozen strong-but-cooling ONI signal."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "calculated"
SOURCE = ROOT.parent / "enso_cop_xle_strong_event_walkforward"
INITIAL_CAPITAL = 100_000.0
RISK_BUDGET = 0.10
OPTION_SLIPPAGE = 0.025
COMMISSION_PER_CONTRACT_SIDE = 0.65
TAX_RATE = 0.37

STRUCTURES = {
    "O1 Long XLE ATM call": [("XLE", "long", 1.0)],
    "O2 Long COP ATM put": [("COP", "long", 1.0)],
    "O3 Long XLE call + COP put": [("XLE", "long", 0.5), ("COP", "long", 0.5)],
    "O4 XLE bull call spread": [("XLE", "vertical", 1.0)],
    "O5 Paired vertical spreads": [("XLE", "vertical", 0.5), ("COP", "vertical", 0.5)],
}


def load_bar(contract: str, date: pd.Timestamp) -> dict:
    payload = json.loads((RAW / f"bars_{contract.replace(':', '_')}.json").read_text())
    by_date = {
        pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize(): r
        for r in payload["results"]
    }
    if date not in by_date:
        raise RuntimeError(f"Missing exact option bar for {contract} on {date.date()}")
    return by_date[date]


def component_prices(selection: pd.DataFrame, underlying: str, kind: str, entry: pd.Timestamp, exit_date: pd.Timestamp) -> dict:
    rows = selection[selection["underlying"] == underlying].set_index("role")
    atm = rows.loc["atm"]
    atm_entry, atm_exit = load_bar(atm["contract"], entry), load_bar(atm["contract"], exit_date)
    if kind == "long":
        raw_entry = float(atm_entry["o"]) * 100
        raw_exit = float(atm_exit["c"]) * 100
        entry_cash = float(atm_entry["o"]) * (1 + OPTION_SLIPPAGE) * 100 + COMMISSION_PER_CONTRACT_SIDE
        exit_cash = float(atm_exit["c"]) * (1 - OPTION_SLIPPAGE) * 100 - COMMISSION_PER_CONTRACT_SIDE
        legs = 1
        contracts_used = atm["contract"]
        entry_volume = int(atm_entry.get("v", 0))
        exit_volume = int(atm_exit.get("v", 0))
    else:
        wing = rows.loc["wing"]
        wing_entry, wing_exit = load_bar(wing["contract"], entry), load_bar(wing["contract"], exit_date)
        raw_entry = (float(atm_entry["o"]) - float(wing_entry["o"])) * 100
        raw_exit = (float(atm_exit["c"]) - float(wing_exit["c"])) * 100
        entry_cash = (
            float(atm_entry["o"]) * (1 + OPTION_SLIPPAGE)
            - float(wing_entry["o"]) * (1 - OPTION_SLIPPAGE)
        ) * 100 + 2 * COMMISSION_PER_CONTRACT_SIDE
        exit_cash = (
            float(atm_exit["c"]) * (1 - OPTION_SLIPPAGE)
            - float(wing_exit["c"]) * (1 + OPTION_SLIPPAGE)
        ) * 100 - 2 * COMMISSION_PER_CONTRACT_SIDE
        legs = 2
        contracts_used = f"{atm['contract']} / {wing['contract']}"
        entry_volume = min(int(atm_entry.get("v", 0)), int(wing_entry.get("v", 0)))
        exit_volume = min(int(atm_exit.get("v", 0)), int(wing_exit.get("v", 0)))
    if entry_cash <= 0:
        raise RuntimeError(f"Non-positive debit for {underlying} {kind}")
    return {
        "underlying": underlying, "component": kind, "contracts_used": contracts_used,
        "entry_cash_per_contract": entry_cash, "exit_cash_per_contract": exit_cash,
        "raw_pnl_per_contract": raw_exit - raw_entry,
        "net_pnl_per_contract": exit_cash - entry_cash,
        "friction_per_contract": (raw_exit - raw_entry) - (exit_cash - entry_cash),
        "legs": legs, "entry_volume_min": entry_volume, "exit_volume_min": exit_volume,
    }


def table(frame: pd.DataFrame, columns: list[tuple[str, str, callable]]) -> str:
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


def bar_chart(summary: pd.DataFrame) -> str:
    width, height, left, right, top, bottom = 1040, 330, 260, 70, 30, 35
    values = summary["after_tax_return"].astype(float)
    bound = max(abs(values.min()), abs(values.max()), 0.01)
    zero = left + (width - left - right) / 2
    usable = (width - left - right) / 2
    row_h = (height - top - bottom) / len(summary)
    items = []
    for i, (_, row) in enumerate(summary.iterrows()):
        value = float(row["after_tax_return"])
        y = top + i * row_h + row_h * 0.18
        bar_w = abs(value) / bound * usable
        x = zero if value >= 0 else zero - bar_w
        color = "#34d399" if value >= 0 else "#fb7185"
        items.append(f"<text x='{left-10}' y='{y+17:.1f}' fill='#d8e4ef' font-size='13' text-anchor='end'>{html.escape(str(row['structure']).split(' ',1)[1])}</text>")
        items.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{row_h*.55:.1f}' rx='3' fill='{color}'/>")
        tx = x + bar_w + 8 if value >= 0 else x - 8
        anchor = "start" if value >= 0 else "end"
        items.append(f"<text x='{tx:.1f}' y='{y+17:.1f}' fill='white' font-size='13' text-anchor='{anchor}'>{value:.1%}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='After-tax option strategy returns'><rect width='100%' height='100%' fill='#0f1d2d'/><line x1='{zero}' x2='{zero}' y1='{top-8}' y2='{height-bottom+4}' stroke='#6b7f95'/>{''.join(items)}</svg>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(ROOT / "contract_selections.csv", parse_dates=["entry_date", "planned_exit_date", "expiration_date"])
    source_trades = pd.read_csv(SOURCE / "calculated" / "non_overlapping_trades.csv", parse_dates=["entry_date", "exit_date"])
    source_trades = source_trades[(source_trades["rule"] == "S5 Strong but cooling") & (source_trades["horizon_days"] == 40)].copy()

    leg_rows = []
    for event_id, group in selections.groupby("event_id"):
        entry = group["entry_date"].iloc[0]
        exit_date = group["planned_exit_date"].iloc[0]
        for underlying, kind in (("XLE", "long"), ("COP", "long"), ("XLE", "vertical"), ("COP", "vertical")):
            row = component_prices(group, underlying, kind, entry, exit_date)
            row.update({"event_id": int(event_id), "entry_date": entry, "exit_date": exit_date})
            leg_rows.append(row)
    components = pd.DataFrame(leg_rows)
    components.to_csv(OUT / "option_component_prices.csv", index=False)

    trade_rows, summary_rows, tax_rows = [], [], []
    for structure, specification in STRUCTURES.items():
        equity = INITIAL_CAPITAL
        pre_tax_profit = 0.0
        raw_profit = 0.0
        friction = 0.0
        deployed = 0.0
        structure_trades = []
        for event_id in sorted(selections["event_id"].unique()):
            event = components[components["event_id"] == event_id]
            event_source = source_trades.iloc[int(event_id) - 1]
            event_deployed = event_pnl = event_raw = event_friction = 0.0
            component_audit = []
            for underlying, kind, weight in specification:
                comp = event[(event["underlying"] == underlying) & (event["component"] == kind)].iloc[0]
                budget = equity * RISK_BUDGET * weight
                contracts = int(math.floor(budget / float(comp["entry_cash_per_contract"])))
                if contracts < 1:
                    raise RuntimeError(f"Budget cannot buy one contract for {structure}")
                comp_deployed = contracts * float(comp["entry_cash_per_contract"])
                comp_pnl = contracts * float(comp["net_pnl_per_contract"])
                comp_raw = contracts * float(comp["raw_pnl_per_contract"])
                comp_friction = contracts * float(comp["friction_per_contract"])
                event_deployed += comp_deployed
                event_pnl += comp_pnl
                event_raw += comp_raw
                event_friction += comp_friction
                component_audit.append(f"{underlying} {kind}: {contracts}x")
            equity_before = equity
            equity += event_pnl
            pre_tax_profit += event_pnl
            raw_profit += event_raw
            friction += event_friction
            deployed += event_deployed
            structure_trades.append(event_pnl)
            trade_rows.append({
                "structure": structure, "event_id": int(event_id),
                "centered_month": event_source["centered_month"], "ONI": event_source["ONI"], "dONI": event_source["dONI"],
                "entry_date": event_source["entry_date"], "exit_date": event_source["exit_date"],
                "components": "; ".join(component_audit), "portfolio_before": equity_before,
                "capital_deployed": event_deployed, "raw_pnl": event_raw, "friction_cost": event_friction,
                "net_pre_tax_pnl": event_pnl, "net_pre_tax_portfolio_return": event_pnl / equity_before,
                "underlying_pair_net_return": event_source["net_pre_tax_return"], "regime": event_source["regime"],
            })
        tax = TAX_RATE * max(pre_tax_profit, 0.0)
        after_tax_profit = pre_tax_profit - tax
        summary_rows.append({
            "structure": structure, "trades": len(structure_trades), "risk_budget_per_trade": RISK_BUDGET,
            "capital_deployed_total": deployed, "raw_profit": raw_profit, "friction_cost": friction,
            "net_pre_tax_profit": pre_tax_profit, "tax_paid": tax, "after_tax_profit": after_tax_profit,
            "gross_portfolio_return": raw_profit / INITIAL_CAPITAL,
            "net_pre_tax_return": pre_tax_profit / INITIAL_CAPITAL,
            "after_tax_return": after_tax_profit / INITIAL_CAPITAL,
            "return_on_deployed_capital": pre_tax_profit / deployed,
            "win_rate": sum(x > 0 for x in structure_trades) / len(structure_trades),
            "worst_trade_pnl": min(structure_trades),
        })
        for rate in (0.00, 0.24, 0.37):
            tax_case = rate * max(pre_tax_profit, 0.0)
            tax_rows.append({"structure": structure, "tax_rate": rate, "after_tax_return": (pre_tax_profit - tax_case) / INITIAL_CAPITAL, "tax_paid": tax_case})
    trades = pd.DataFrame(trade_rows)
    summary = pd.DataFrame(summary_rows).sort_values("after_tax_return", ascending=False).reset_index(drop=True)
    tax_sensitivity = pd.DataFrame(tax_rows)
    trades.to_csv(OUT / "option_strategy_trades.csv", index=False)
    summary.to_csv(OUT / "option_strategy_summary.csv", index=False)
    tax_sensitivity.to_csv(OUT / "option_tax_sensitivity.csv", index=False)

    underlying = pd.read_csv(SOURCE / "calculated" / "strategy_summary.csv")
    underlying = underlying[(underlying["rule"] == "S5 Strong but cooling") & (underlying["horizon_days"] == 40)].iloc[0]
    comparison = pd.DataFrame([
        {"implementation": "Underlying long XLE / short COP", "capital_at_risk": INITIAL_CAPITAL, "trades": int(underlying["trades"]), "net_pre_tax_return": underlying["net_pre_tax_cumulative_return"], "after_tax_return": underlying["after_tax_cumulative_return"]},
        *[{"implementation": row["structure"], "capital_at_risk": row["capital_deployed_total"], "trades": int(row["trades"]), "net_pre_tax_return": row["net_pre_tax_return"], "after_tax_return": row["after_tax_return"]} for _, row in summary.iterrows()],
    ])
    comparison.to_csv(OUT / "underlying_option_comparison.csv", index=False)

    hashes = {}
    files = [ROOT / "contract_selections.csv", SOURCE / "calculated" / "non_overlapping_trades.csv"] + list(RAW.glob("bars_O_*.json"))
    for path in files:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "massive_contract_endpoint": "/v3/reference/options/contracts",
        "massive_bar_endpoint": "/v2/aggs/ticker/{optionsTicker}/range/1/day/{from}/{to}",
        "signal": "S5 Strong but cooling, 40 trading days",
        "initial_capital": INITIAL_CAPITAL, "risk_budget_per_trade": RISK_BUDGET,
        "option_slippage_each_transaction": OPTION_SLIPPAGE,
        "commission_per_contract_per_side": COMMISSION_PER_CONTRACT_SIDE,
        "short_term_tax_rate": TAX_RATE, "hashes": hashes,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    best = summary.iloc[0]
    best_trades = trades[trades["structure"] == best["structure"]]
    chart = bar_chart(summary)
    pct = lambda x: "n.a." if pd.isna(x) else f"{100 * float(x):.2f}%"
    usd = lambda x: "n.a." if pd.isna(x) else f"${float(x):,.0f}"
    integer = lambda x: str(int(x))
    selections_show = selections.copy()
    selections_show["dte"] = (selections_show["expiration_date"] - selections_show["entry_date"]).dt.days

    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ONI COP/XLE option strategy adaptation</title><style>
:root{{--card:#101e2e;--text:#eef5fb;--muted:#a6b5c7;--line:#2b3d52;--blue:#38bdf8;--amber:#fbbf24;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07101b,#0b1928);color:var(--text);font:15px/1.58 Inter,system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:44px 24px 72px}}h1{{font-size:42px;line-height:1.08;letter-spacing:-.03em;max-width:940px}}h2{{font-size:25px;margin-top:44px}}h3{{font-size:18px;margin-top:28px}}p,li{{max-width:950px}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:760;letter-spacing:.14em;text-transform:uppercase}}.lede{{font-size:19px;color:#cad7e5}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px}}.metric{{font-size:27px;font-weight:760}}.label{{color:var(--muted);font-size:13px}}.callout{{border-left:4px solid var(--amber);padding:14px 18px;background:#1d1d20;border-radius:0 9px 9px 0;margin:18px 0}}.warning{{border-left-color:var(--red)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:14px 0 24px}}table{{border-collapse:collapse;width:100%;min-width:900px;background:var(--card)}}th{{text-align:left;background:#162a40;color:#c6d7e9;font-size:12px;text-transform:uppercase}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}svg{{display:block;width:100%;max-width:1080px;border:1px solid var(--line);border-radius:12px;margin:18px 0}}code{{background:#142438;padding:2px 5px;border-radius:4px}}a{{color:#7dd3fc}}footer{{border-top:1px solid var(--line);margin-top:44px;padding-top:18px;color:var(--muted)}}@media(max-width:820px){{.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}}}@media(max-width:520px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><div class='eyebrow'>Historical Massive option bars · frozen ONI signal</div><h1>Option implementations of the strong-but-cooling ONI trade</h1><p class='lede'>The underlying strategy called for long XLE and short COP for 40 trading days after delayed ONI remained at least 1.0 but started declining. This report freezes the same two 2024 signals and compares five defined-risk option structures.</p>
<div class='cards'><div class='card'><div class='metric'>{html.escape(str(best['structure']).split(' ',1)[1])}</div><div class='label'>Highest after-tax option implementation</div></div><div class='card'><div class='metric'>{pct(best['after_tax_return'])}</div><div class='label'>After-tax return on the $100,000 portfolio</div></div><div class='card'><div class='metric'>{pct(best['return_on_deployed_capital'])}</div><div class='label'>Pre-tax return on option debit deployed</div></div><div class='card'><div class='metric'>{pct(underlying['after_tax_cumulative_return'])}</div><div class='label'>Underlying pair after-tax return</div></div></div>
<div class='callout warning'><b>Evidence limit.</b> Only two non-overlapping trades exist. Ranking five option structures on those same trades is ex-post selection and cannot establish which structure will win in another El Niño cycle.</div>
<h2>Result</h2><p><b>{html.escape(str(best['structure']))}</b> ranked first. It generated {usd(best['net_pre_tax_profit'])} before tax and after modeled trading friction, or {pct(best['net_pre_tax_return'])} of initial portfolio capital. The 37% short-term tax scenario reduces that to {usd(best['after_tax_profit'])}, or {pct(best['after_tax_return'])}. At most 10% of current portfolio equity was allocated to option debit at each signal.</p><div class='callout warning'><b>Liquidity warning.</b> The winning COP puts had reported exit-day volume of only 8 and 2 contracts. The closing trade marks may not represent executable prices for the simulated 23-contract exits, even after the 2.5% price haircut.</div>{chart}
<h2>Five option structures</h2>{table(summary,[('structure','Structure',str),('trades','Trades',integer),('capital_deployed_total','Total debit deployed',usd),('raw_profit','P&L before friction',usd),('friction_cost','Modeled friction',usd),('net_pre_tax_profit','Net pre-tax P&L',usd),('gross_portfolio_return','Before-friction return',pct),('net_pre_tax_return','Net pre-tax return',pct),('after_tax_return','After-tax return',pct),('return_on_deployed_capital','Return on debit',pct),('win_rate','Win rate',pct)])}
<h2>Trade audit for the highest-ranked structure</h2>{table(best_trades,[('centered_month','Centered ONI month',str),('ONI','ONI',lambda x:f'{float(x):.1f}'),('dONI','ΔONI',lambda x:f'{float(x):+.1f}'),('entry_date','Entry',lambda x:str(pd.Timestamp(x).date())),('exit_date','Exit',lambda x:str(pd.Timestamp(x).date())),('components','Contracts',str),('capital_deployed','Debit deployed',usd),('raw_pnl','P&L before friction',usd),('friction_cost','Friction',usd),('net_pre_tax_pnl','Net P&L',usd),('net_pre_tax_portfolio_return','Portfolio return',pct),('regime','Entry regime',str)])}
<h2>Contract selection</h2><p>Contracts were queried with Massive's historical <code>as_of</code> parameter at each entry. Expiry had to be at least seven calendar days beyond the planned exit and closest to 75 days from entry. The at-the-money contract minimizes strike distance to the unadjusted underlying opening price. The vertical wing targets approximately 10% out of the money. Every selected contract has an option trade bar on both the exact entry and exit date.</p>{table(selections_show,[('event_id','Event',integer),('entry_date','Entry',lambda x:str(pd.Timestamp(x).date())),('underlying','Underlying',str),('option_type','Type',str),('role','Role',str),('spot_open','Spot open',lambda x:f'${float(x):.2f}'),('contract','Contract',str),('strike','Strike',lambda x:f'${float(x):.2f}'),('expiration_date','Expiry',lambda x:str(pd.Timestamp(x).date())),('dte','Entry DTE',integer)])}
<h3>Option price and liquidity audit</h3>{table(components,[('event_id','Event',integer),('underlying','Underlying',str),('component','Component',str),('entry_cash_per_contract','Modeled entry debit',usd),('exit_cash_per_contract','Modeled exit value',usd),('net_pnl_per_contract','Net P&L per contract',usd),('friction_per_contract','Friction per contract',usd),('entry_volume_min','Entry-day minimum volume',integer),('exit_volume_min','Exit-day minimum volume',integer)])}
<h2>Underlying comparison</h2>{table(comparison,[('implementation','Implementation',str),('trades','Trades',integer),('capital_at_risk','Capital/debit measure',usd),('net_pre_tax_return','Net pre-tax return',pct),('after_tax_return','After-tax return',pct)])}<p>Option and stock percentages carry different exposure. The underlying pair uses 100% gross exposure split between XLE and COP. Options risk only a 10% debit budget per signal, with the remaining portfolio held as zero-yield cash. Return on option debit is shown separately.</p>
<h2>Execution and tax assumptions</h2><ul><li>Entry uses the option's first reported trade-bar open on the signal date. Exit uses the same contract's reported close on the planned 40th trading day.</li><li>Each option purchase is marked 2.5% above the trade price and each sale 2.5% below it. Short spread legs receive 2.5% less on entry and cost 2.5% more to close.</li><li>Commission is $0.65 per contract per leg at both entry and exit.</li><li>Contract quantity is the largest whole number whose modeled entry debit remains inside its assigned 10% risk budget.</li><li>The base tax scenario applies 37% to positive aggregate 2024 short-term profit after friction. Losses offset gains inside the test year. State tax, net-investment-income tax, loss carryforwards and cash yield are excluded.</li></ul>
<h3>Tax sensitivity</h3>{table(tax_sensitivity,[('structure','Structure',str),('tax_rate','Tax rate',pct),('after_tax_return','After-tax return',pct),('tax_paid','Tax paid',usd)])}
<h2>What the option result measures</h2><p>The test measures realized option-price P&amp;L, not signal quality. Signal quality was established by the earlier walk-forward COP-minus-XLE return calculation. Option P&amp;L adds strike selection, expiry, convexity, time decay, volatility pricing and liquidity. A correct underlying direction can still lose money in a long option if the move is too small or arrives too late.</p>
<h2>Limitations</h2><ul><li>Massive daily option aggregates contain qualifying trades, not historical bid-ask quotes. The 2.5% execution haircut is a scenario assumption and cannot reconstruct the actual spread.</li><li>Historical implied volatility and Greeks were not available from the daily-bar files and were not backfilled from current snapshots.</li><li>Daily option closes can reflect trades at different times. Requiring exact entry and exit bars reduces stale-mark risk but does not synchronize every leg.</li><li>The two signals occurred in the same 2023–24 El Niño episode and the same energy uptrend/low-volatility regime.</li><li>The best structure was selected from five candidates after observing both trades.</li></ul>
<h2>Sources</h2><ul><li><a href='https://massive.com/docs/rest/options/contracts/all-contracts'>Massive option-contract reference</a></li><li><a href='https://massive.com/docs/rest/options/aggregates/custom-bars'>Massive historical option aggregate bars</a></li><li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/'>NOAA CPC ONI table and revision note</a></li><li><a href='https://www.irs.gov/taxtopics/tc409'>IRS Topic 409: short-term gains</a></li></ul>
<footer>Initial capital: ${INITIAL_CAPITAL:,.0f}. Maximum option debit per signal: {RISK_BUDGET:.0%}. Option-price haircut: {OPTION_SLIPPAGE:.1%} each transaction. All contract selections, component prices, trades, summaries and source hashes are saved beside this report.</footer></main></body></html>"""
    (ROOT / "ONI_COP_XLE_options_adaptation_report.html").write_text(report)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
