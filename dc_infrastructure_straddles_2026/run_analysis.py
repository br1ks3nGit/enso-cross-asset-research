#!/usr/bin/env python3
"""Analyze 2026 monthly ATM straddles on data-center infrastructure equities."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "calculated"
ONI_SOURCE = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(ROOT.parent / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
TICKERS = ("VRT", "MOD", "ETN", "PWR", "GEV")
INITIAL_CAPITAL_HKD = 100_000.0
OPTION_SLIPPAGE_EACH_SIDE = 0.025
COMMISSION_USD_PER_CONTRACT_PER_LEG_PER_SIDE = 0.65
REFERENCE_RISK_CAP = 0.20
ONI_PUBLICATION_DELAY_MONTHS = 3

COMPANY_ROLES = {
    "VRT": "Power and thermal infrastructure",
    "MOD": "Thermal-management systems",
    "ETN": "Electrical power management",
    "PWR": "Grid and electrical infrastructure construction",
    "GEV": "Power generation and grid equipment",
}


def load_underlying(ticker: str) -> pd.DataFrame:
    payload = json.loads((RAW / f"massive_{ticker}_daily.json").read_text())
    rows = payload["results"]
    return pd.DataFrame({
        "date": [pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize() for r in rows],
        "open": [float(r["o"]) for r in rows],
        "close": [float(r["c"]) for r in rows],
    }).set_index("date").sort_index()


def load_signals() -> pd.DataFrame:
    oni = pd.read_csv(ONI_SOURCE, parse_dates=["Date"])[["Date", "ONI", "dONI"]].copy()
    oni["centered_month"] = oni["Date"].dt.to_period("M")
    rows = []
    for month in pd.period_range("2026-04", "2026-09", freq="M"):
        # Conservative proxy: at the first session of month M, use M-3 centered ONI.
        available_month = month - ONI_PUBLICATION_DELAY_MONTHS
        match = oni[oni["centered_month"] == available_month]
        if match.empty:
            raise RuntimeError(f"Missing ONI for {available_month}")
        row = match.iloc[0]
        strength = "Strong (≥1.0)" if row["ONI"] >= 1.0 else "Moderate (0.5–0.9)" if row["ONI"] >= 0.5 else "Below El Niño threshold"
        rows.append({
            "decision_month": str(month),
            "available_centered_month": str(available_month),
            "ONI": float(row["ONI"]),
            "dONI": float(row["dONI"]),
            "oni_gate": bool(row["ONI"] >= 0.5 and row["dONI"] > 0),
            "signal_strength": strength,
        })
    return pd.DataFrame(rows)


def fx_value(fx: pd.DataFrame, date: pd.Timestamp, column: str) -> float:
    exact = fx[fx["date"] == date]
    if exact.empty:
        prior = fx[fx["date"] <= date]
        if prior.empty:
            raise RuntimeError(f"No USD/HKD observation on or before {date.date()}")
        return float(prior.iloc[-1][column])
    return float(exact.iloc[0][column])


def prepare_trades() -> pd.DataFrame:
    selections = pd.read_csv(
        ROOT / "contract_selections.csv",
        parse_dates=["entry_date", "exit_date", "expiration_date"],
    )
    signals = load_signals()
    fx = pd.read_csv(ROOT / "usd_hkd_daily.csv", parse_dates=["date"])
    underlyings = {ticker: load_underlying(ticker) for ticker in TICKERS}
    rows = []
    for _, selection in selections.iterrows():
        ticker = selection["ticker"]
        entry = selection["entry_date"]
        exit_date = selection["exit_date"]
        signal = signals[signals["decision_month"] == str(entry.to_period("M"))].iloc[0]
        fx_entry = fx_value(fx, entry, "open")
        fx_exit = fx_value(fx, exit_date, "close")
        call_entry = float(selection["call_entry_open"])
        put_entry = float(selection["put_entry_open"])
        call_exit = float(selection["call_exit_close"])
        put_exit = float(selection["put_exit_close"])
        raw_entry_usd = (call_entry + put_entry) * 100
        raw_exit_usd = (call_exit + put_exit) * 100
        commission_side = 2 * COMMISSION_USD_PER_CONTRACT_PER_LEG_PER_SIDE
        entry_debit_usd = raw_entry_usd * (1 + OPTION_SLIPPAGE_EACH_SIDE) + commission_side
        exit_proceeds_usd = max(raw_exit_usd * (1 - OPTION_SLIPPAGE_EACH_SIDE) - commission_side, 0.0)
        raw_entry_hkd = raw_entry_usd * fx_entry
        raw_exit_hkd = raw_exit_usd * fx_exit
        entry_debit_hkd = entry_debit_usd * fx_entry
        exit_proceeds_hkd = exit_proceeds_usd * fx_exit
        net_pnl_hkd = exit_proceeds_hkd - entry_debit_hkd
        raw_pnl_hkd = raw_exit_hkd - raw_entry_hkd
        underlying_entry = float(underlyings[ticker].loc[entry, "open"])
        underlying_exit = float(underlyings[ticker].loc[exit_date, "close"])
        underlying_return_usd = underlying_exit / underlying_entry - 1
        underlying_return_hkd = (underlying_exit * fx_exit) / (underlying_entry * fx_entry) - 1
        premium_implied_move = (call_entry + put_entry) / underlying_entry
        rows.append({
            **selection.to_dict(),
            "company_role": COMPANY_ROLES[ticker],
            "available_centered_month": signal["available_centered_month"],
            "ONI": signal["ONI"],
            "dONI": signal["dONI"],
            "oni_gate": signal["oni_gate"],
            "signal_strength": signal["signal_strength"],
            "fx_entry_hkd_per_usd": fx_entry,
            "fx_exit_hkd_per_usd": fx_exit,
            "raw_entry_premium_usd": raw_entry_usd,
            "raw_exit_value_usd": raw_exit_usd,
            "entry_debit_hkd": entry_debit_hkd,
            "exit_proceeds_hkd": exit_proceeds_hkd,
            "raw_pnl_hkd": raw_pnl_hkd,
            "friction_cost_hkd": raw_pnl_hkd - net_pnl_hkd,
            "net_pnl_hkd": net_pnl_hkd,
            "net_return_on_debit": net_pnl_hkd / entry_debit_hkd,
            "within_20pct_initial_cap": entry_debit_hkd <= INITIAL_CAPITAL_HKD * REFERENCE_RISK_CAP,
            "affordable_at_initial_capital": entry_debit_hkd <= INITIAL_CAPITAL_HKD,
            "underlying_entry": underlying_entry,
            "underlying_exit": underlying_exit,
            "underlying_return_usd": underlying_return_usd,
            "underlying_return_hkd": underlying_return_hkd,
            "abs_underlying_return": abs(underlying_return_usd),
            "premium_implied_move": premium_implied_move,
            "magnitude_coverage": abs(underlying_return_usd) / premium_implied_move,
            "minimum_leg_volume": min(
                int(selection["call_entry_volume"]), int(selection["put_entry_volume"]),
                int(selection["call_exit_volume"]), int(selection["put_exit_volume"]),
            ),
        })
    frame = pd.DataFrame(rows).sort_values(["ticker", "entry_date"]).reset_index(drop=True)
    frame["option_win"] = frame["net_pnl_hkd"] > 0
    frame["liquidity_flag"] = frame["minimum_leg_volume"].map(lambda x: "Fragile (≤5)" if x <= 5 else "Better (>5)")
    return frame


def simulate(frame: pd.DataFrame, gated: bool, completed_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = frame.copy()
    if gated:
        candidates = candidates[candidates["oni_gate"]]
    if completed_only:
        candidates = candidates[~candidates["partial_window"]]
    trade_rows = []
    summary_rows = []
    for ticker in TICKERS:
        equity = INITIAL_CAPITAL_HKD
        executed = 0
        skipped = 0
        for _, trade in candidates[candidates["ticker"] == ticker].sort_values("entry_date").iterrows():
            equity_before = equity
            affordable = float(trade["entry_debit_hkd"]) <= equity
            if affordable:
                pnl = float(trade["net_pnl_hkd"])
                equity += pnl
                executed += 1
                status = "Executed: one straddle"
            else:
                pnl = 0.0
                skipped += 1
                status = "Skipped: one straddle cost exceeded equity"
            trade_rows.append({
                "universe": "ONI-gated" if gated else "Monthly",
                "sample": "Completed only" if completed_only else "Through 2026-09-11",
                "ticker": ticker,
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "status": status,
                "equity_before_hkd": equity_before,
                "entry_debit_hkd": trade["entry_debit_hkd"],
                "realized_pnl_hkd": pnl,
                "equity_after_hkd": equity,
                "risk_share_if_executed": trade["entry_debit_hkd"] / equity_before,
                "within_20pct_risk_cap": trade["entry_debit_hkd"] <= equity_before * REFERENCE_RISK_CAP,
                "partial_window": trade["partial_window"],
            })
        summary_rows.append({
            "universe": "ONI-gated" if gated else "Monthly",
            "sample": "Completed only" if completed_only else "Through 2026-09-11",
            "ticker": ticker,
            "candidate_trades": len(candidates[candidates["ticker"] == ticker]),
            "executed_trades": executed,
            "skipped_unaffordable": skipped,
            "ending_equity_hkd": equity,
            "net_profit_hkd": equity - INITIAL_CAPITAL_HKD,
            "portfolio_return": equity / INITIAL_CAPITAL_HKD - 1,
        })
    return pd.DataFrame(trade_rows), pd.DataFrame(summary_rows)


def signal_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        subset = frame[frame["ticker"] == ticker]
        for label, group in (("ONI gate true", subset[subset["oni_gate"]]), ("ONI gate false", subset[~subset["oni_gate"]])):
            rows.append({
                "ticker": ticker,
                "signal_group": label,
                "observations": len(group),
                "mean_abs_underlying_return": group["abs_underlying_return"].mean(),
                "median_abs_underlying_return": group["abs_underlying_return"].median(),
                "mean_magnitude_coverage": group["magnitude_coverage"].mean(),
                "mean_net_straddle_return": group["net_return_on_debit"].mean(),
                "straddle_win_rate": group["option_win"].mean(),
            })
    return pd.DataFrame(rows)


def benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        prices = load_underlying(ticker)
        entry_date = prices.index.min()
        exit_date = prices.index.max()
        fx = pd.read_csv(ROOT / "usd_hkd_daily.csv", parse_dates=["date"])
        fx_entry = fx_value(fx, entry_date, "open")
        fx_exit = fx_value(fx, exit_date, "close")
        ret = (float(prices.loc[exit_date, "close"]) * fx_exit) / (float(prices.loc[entry_date, "open"]) * fx_entry) - 1
        rows.append({"ticker": ticker, "entry_date": entry_date, "exit_date": exit_date, "buy_hold_return_hkd": ret})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "ticker": "Equal-weight five-name benchmark",
        "entry_date": out["entry_date"].min(),
        "exit_date": out["exit_date"].max(),
        "buy_hold_return_hkd": out["buy_hold_return_hkd"].mean(),
    }
    return out


def shadow_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One contract on every observed row, ignoring account affordability."""
    return frame.groupby("ticker", as_index=False).agg(
        observed_straddles=("ticker", "size"),
        raw_pnl_hkd=("raw_pnl_hkd", "sum"),
        friction_cost_hkd=("friction_cost_hkd", "sum"),
        net_pnl_hkd=("net_pnl_hkd", "sum"),
        mean_net_return_on_debit=("net_return_on_debit", "mean"),
        wins=("option_win", "sum"),
        mean_entry_debit_hkd=("entry_debit_hkd", "mean"),
    ).sort_values("net_pnl_hkd", ascending=False)


def fmt_table(frame: pd.DataFrame, columns: list[tuple[str, str, object]]) -> str:
    heads = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row.get(key)
            text = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def result_chart(summary: pd.DataFrame) -> str:
    values = summary.set_index("ticker")["portfolio_return"].reindex(TICKERS)
    width, height, left, right, top, bottom = 920, 320, 100, 30, 35, 52
    bound = max(abs(values.min()), abs(values.max()), 0.05)
    zero = top + (height - top - bottom) / 2
    usable = (height - top - bottom) / 2
    bar_w = (width - left - right) / len(values) * 0.58
    gap = (width - left - right) / len(values)
    parts = [f"<line x1='{left}' x2='{width-right}' y1='{zero}' y2='{zero}' stroke='#718096'/>" ]
    for i, (ticker, value) in enumerate(values.items()):
        x = left + i * gap + (gap - bar_w) / 2
        h = abs(float(value)) / bound * usable
        y = zero - h if value >= 0 else zero
        color = "#34d399" if value >= 0 else "#fb7185"
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' rx='4' fill='{color}'/>")
        ty = y - 8 if value >= 0 else y + h + 18
        parts.append(f"<text x='{x+bar_w/2:.1f}' y='{ty:.1f}' text-anchor='middle' fill='#edf6ff' font-size='13'>{value:.1%}</text>")
        parts.append(f"<text x='{x+bar_w/2:.1f}' y='{height-20}' text-anchor='middle' fill='#c7d6e5' font-size='14'>{ticker}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Capital-constrained straddle returns'><rect width='100%' height='100%' fill='#0f1d2d'/>{''.join(parts)}</svg>"


def build_report(
    trades: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    gated_summary: pd.DataFrame,
    completed_summary: pd.DataFrame,
    quality: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    shadow: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    pct = lambda x: "n.a." if pd.isna(x) else f"{100 * float(x):.2f}%"
    hkd = lambda x: "n.a." if pd.isna(x) else f"HK${float(x):,.0f}"
    num = lambda x: "n.a." if pd.isna(x) else f"{float(x):.2f}"
    integer = lambda x: str(int(x))
    boolfmt = lambda x: "Yes" if bool(x) else "No"
    datefmt = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")

    ranked = monthly_summary.sort_values("portfolio_return", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    fully_observed = ranked[(ranked["candidate_trades"] == 5) & (ranked["executed_trades"] == 5)]
    best_full = fully_observed.sort_values("portfolio_return", ascending=False).iloc[0]
    completed_ranked = completed_summary.sort_values("portfolio_return", ascending=False).reset_index(drop=True)
    available = len(trades)
    total_possible = len(TICKERS) * 6
    risk_cap_count = int(trades["within_20pct_initial_cap"].sum())
    affordable_count = int(trades["affordable_at_initial_capital"].sum())
    fragile = int((trades["minimum_leg_volume"] <= 5).sum())
    chart = result_chart(ranked)

    ranking_table = fmt_table(ranked, [
        ("ticker", "Ticker", None),
        ("candidate_trades", "Candidates", integer),
        ("executed_trades", "Executed", integer),
        ("skipped_unaffordable", "Skipped", integer),
        ("ending_equity_hkd", "Ending equity", hkd),
        ("net_profit_hkd", "Net P&L", hkd),
        ("portfolio_return", "Return", pct),
    ])
    completed_table = fmt_table(completed_ranked, [
        ("ticker", "Ticker", None), ("executed_trades", "Executed", integer),
        ("net_profit_hkd", "Net P&L", hkd), ("portfolio_return", "Return", pct),
    ])
    gated_table = fmt_table(gated_summary.sort_values("portfolio_return", ascending=False), [
        ("ticker", "Ticker", None), ("candidate_trades", "Gated candidates", integer),
        ("executed_trades", "Executed", integer), ("skipped_unaffordable", "Skipped", integer),
        ("net_profit_hkd", "Net P&L", hkd), ("portfolio_return", "Return", pct),
    ])
    trade_table = fmt_table(trades.sort_values(["entry_date", "ticker"]), [
        ("ticker", "Ticker", None), ("entry_date", "Entry", datefmt), ("exit_date", "Exit", datefmt),
        ("partial_window", "Partial?", boolfmt), ("available_centered_month", "ONI month available", None),
        ("ONI", "ONI", num), ("oni_gate", "Gate?", boolfmt), ("strike", "Strike", num),
        ("dte_at_entry", "DTE", integer), ("entry_debit_hkd", "1-straddle debit", hkd),
        ("net_pnl_hkd", "Net P&L", hkd), ("net_return_on_debit", "Return/debit", pct),
        ("abs_underlying_return", "|Stock move|", pct), ("premium_implied_move", "Premium/spot", pct),
        ("magnitude_coverage", "Magnitude coverage", num), ("minimum_leg_volume", "Min leg volume", integer),
    ])
    quality_table = fmt_table(quality, [
        ("ticker", "Ticker", None), ("signal_group", "Signal group", None),
        ("observations", "n", integer), ("mean_abs_underlying_return", "Mean |stock move|", pct),
        ("mean_magnitude_coverage", "Mean magnitude coverage", num),
        ("mean_net_straddle_return", "Mean straddle return", pct), ("straddle_win_rate", "Win rate", pct),
    ])
    shadow_table = fmt_table(shadow, [
        ("ticker", "Ticker", None), ("observed_straddles", "Observed", integer),
        ("mean_entry_debit_hkd", "Mean 1-straddle debit", hkd),
        ("raw_pnl_hkd", "Raw P&L", hkd), ("friction_cost_hkd", "Modeled friction", hkd),
        ("net_pnl_hkd", "Net P&L", hkd), ("mean_net_return_on_debit", "Mean return/debit", pct),
        ("wins", "Wins", integer),
    ])
    benchmark_table = fmt_table(benchmark_frame, [
        ("ticker", "Benchmark", None), ("entry_date", "Start", datefmt),
        ("exit_date", "End", datefmt), ("buy_hold_return_hkd", "HKD buy-and-hold return", pct),
    ])
    failures_table = fmt_table(failures, [
        ("ticker", "Ticker", None), ("entry_date", "Entry", datefmt),
        ("exit_date", "Planned exit", datefmt), ("reason", "Reason", None),
    ]) if not failures.empty else "<p>No selection failures.</p>"

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Data-center infrastructure long-straddle backtest</title><style>
:root{{--bg:#07111d;--card:#0f1d2d;--text:#eef6fd;--muted:#a8b7c8;--line:#2b4055;--blue:#38bdf8;--green:#34d399;--amber:#fbbf24;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#06101a,#0a1928);color:var(--text);font:15px/1.58 Inter,system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:46px 24px 76px}}h1{{font-size:42px;line-height:1.08;letter-spacing:-.035em;max-width:1000px}}h2{{font-size:26px;margin-top:46px}}h3{{font-size:19px;margin-top:30px}}p,li{{max-width:980px}}a{{color:#7dd3fc}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:760;letter-spacing:.14em;text-transform:uppercase}}.lede{{font-size:19px;color:#cedae7}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}.metric{{font-size:27px;font-weight:760}}.label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}.callout{{border-left:4px solid var(--amber);background:#172131;padding:15px 18px;border-radius:8px;margin:22px 0}}.good{{border-left-color:var(--green)}}.bad{{border-left-color:var(--red)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:15px 0 25px}}table{{border-collapse:collapse;width:100%;min-width:760px;background:#0c1927}}th,td{{padding:10px 12px;border-bottom:1px solid #22364a;text-align:right;white-space:nowrap}}th{{color:#b9c9d8;background:#112337;font-size:12px}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;margin:10px 0 20px}}code{{background:#12263a;padding:2px 5px;border-radius:4px}}.muted{{color:var(--muted)}}footer{{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}}}
</style></head><body><main><div class='eyebrow'>Massive point-in-time option bars · through 11 September 2026</div><h1>Long straddles found volatility—but HKD 100,000 is too small for prudent sizing</h1><p class='lede'>Monthly, same-strike ATM call-plus-put positions on VRT, MOD, ETN, PWR and GEV. Each company is simulated as a separate HKD 100,000 account. The primary capital-constrained rule buys exactly one straddle only when its all-in debit fits available cash; it never uses leverage or fractional contracts.</p>
<div class='cards'><div class='card'><div class='label'>Least loss; only 1/5 executed</div><div class='metric'>{best['ticker']} {pct(best['portfolio_return'])}</div></div><div class='card'><div class='label'>Best with all 5 executed</div><div class='metric'>{best_full['ticker']} {pct(best_full['portfolio_return'])}</div></div><div class='card'><div class='label'>Point-in-time pairs found</div><div class='metric'>{available}/{total_possible}</div></div><div class='card'><div class='label'>Within 20% risk cap</div><div class='metric'>{risk_cap_count}/{available}</div></div></div>
<div class='callout bad'><strong>Decision:</strong> this is not a well-sized retail strategy at HKD 100,000. Zero selected straddles fit a 20% premium-at-risk cap. The one-contract simulation is mechanically cash-funded, but often puts a very large share of the account at risk. Treat its ranking as research evidence, not an implementable recommendation.</div>
<h2>Capital-constrained results</h2><p>Returns include USD/HKD conversion at each entry and exit, 2.5% option-price slippage on every transaction, and USD 0.65 commission per contract, per leg, per side. Base-case tax is zero because Hong Kong generally does not tax capital gains; classification as taxable trading income is facts-dependent and is not resolved by this model.</p><p><strong>GEV's −3.04% is not evidence that it was the best straddle.</strong> Four of five candidate GEV trades were unaffordable and therefore contributed zero. Among names with all five available trades executed, VRT lost 28.59% and ETN lost 63.04%.</p>{chart}{ranking_table}
<p class='muted'>The September positions are marked to market on 11 September after only eight trading sessions. They are not annualized or extrapolated.</p>
<h3>Completed 20-session windows only</h3>{completed_table}
<h3>One-contract shadow result</h3><p>This view buys one straddle on every observed ticker-month even when the HKD 100,000 account could not afford it. It is not an executable portfolio, but it removes the misleading benefit of skipped trades. Raw P&amp;L before modeled slippage and commissions was negative for every ticker, so transaction-cost assumptions are not the sole reason the strategy failed.</p>{shadow_table}
<h2>Does delayed ONI improve the selection?</h2><p>The ONI gate is true when the latest conservatively delayed centered observation is at least +0.5 and still rising. At the first session of month M, the model uses centered month M−3. This removes obvious look-ahead, but it is a publication-delay proxy—not a true NOAA vintage archive.</p>{gated_table}
<h3>Signal quality, separate from P&amp;L</h3><p><strong>Magnitude coverage</strong> is |stock return| divided by entry straddle premium/spot. It measures whether the stock move was large relative to the option premium’s simple expiry breakeven proxy. It is not an option return and does not account for residual time value.</p>{quality_table}
<div class='callout'><strong>Interpretation discipline:</strong> ONI-gated rows are July–September only in this tiny sample. Differences can be caused by earnings, AI-capex news, rates or company events. With at most three gated observations per liquid ticker, no p-value would be informative; the report deliberately avoids claiming statistical significance.</div>
<h2>Trade-level audit</h2>{trade_table}
<p>{fragile} of {available} selected ticker-months have minimum observed volume of five contracts or fewer across their four entry/exit leg bars. A daily aggregate proves a trade occurred, not that a two-leg order of arbitrary size was executable at the reported open/close.</p>
<h2>Underlying benchmark</h2><p>This benchmark buys each stock at its 1 April open and marks it at the 11 September close, translating both cash flows through Massive USD/HKD bars. It is a directional comparator, not a volatility strategy.</p>{benchmark_table}
<h2>Missing selections</h2><p>Massive returned no contracts in the 45–70 DTE point-in-time chain on 1 April for all five names. MOD also lacked a same-strike pair with exact entry and exit trade bars in August and September. Missing rows remain missing.</p>{failures_table}
<h2>Exact formulas and metric definitions</h2><ul>
<li><strong>Raw entry premium (USD)</strong> = 100 × (call open + put open).</li>
<li><strong>All-in entry debit (USD)</strong> = raw entry premium × (1 + 2.5%) + 2 × USD 0.65.</li>
<li><strong>Exit proceeds (USD)</strong> = max[100 × (call close + put close) × (1 − 2.5%) − 2 × USD 0.65, 0].</li>
<li><strong>Net P&amp;L (HKD)</strong> = exit proceeds USD × exit USD/HKD − entry debit USD × entry USD/HKD.</li>
<li><strong>Return on debit</strong> = net P&amp;L HKD ÷ entry debit HKD.</li>
<li><strong>Portfolio return</strong> = (ending HKD equity ÷ HKD 100,000) − 1. Skipped trades contribute zero.</li>
<li><strong>Premium/spot</strong> = (call open + put open) ÷ stock open. It approximates the expiry move needed to cover premium before costs.</li>
<li><strong>Absolute stock move</strong> = |stock exit close ÷ stock entry open − 1|.</li>
<li><strong>Magnitude coverage</strong> = absolute stock move ÷ premium/spot. Above 1.0 means the underlying move exceeded that simple premium ratio; it does not guarantee option profit before expiry.</li>
<li><strong>Win rate</strong> = count(net straddle P&amp;L &gt; 0) ÷ observed straddles.</li>
</ul>
<h2>Method and limitations</h2><ol><li>Entry is the first common U.S. trading session of each month from April through September 2026.</li><li>At entry, query Massive’s contract reference endpoint with that date in <code>as_of</code>; select a common call/put expiry 45–70 days out, closest to 55 DTE, then the same strike nearest the stock open.</li><li>Require an exact option trade bar for both legs on entry and exit. Exit is the twentieth trading session including entry; September stops on the latest available date.</li><li>No bid/ask quotes were available in this dataset. The 2.5% per-transaction haircut is a sensitivity assumption, not a reconstruction of the NBBO.</li><li>Option opens/closes can be stale or unrepresentative in thin contracts. Corporate events, early exercise, dividends and tax-lot rules are not modeled.</li><li>The company universe was economically selected before return ranking. It is still a small, non-random sample with substantial multiple-testing risk.</li></ol>
<h2>Sources</h2><ul><li><a href='https://massive.com/docs/rest/options/contracts/all-contracts'>Massive option contract reference</a></li><li><a href='https://massive.com/docs/rest/options/aggregates/custom-bars'>Massive option aggregate bars</a></li><li><a href='https://massive.com/docs/rest/forex/aggregates/custom-bars'>Massive USD/HKD aggregate bars</a></li><li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/'>NOAA CPC ONI table and revision note</a></li><li><a href='https://www.ird.gov.hk/eng/pdf/pam61e.pdf'>Hong Kong IRD: taxation of financial assets</a></li></ul>
<footer>Starting capital: HKD {INITIAL_CAPITAL_HKD:,.0f} per standalone company simulation. Test span: 1 April–11 September 2026. Raw Massive responses, selected contracts, calculated trades, summaries and hashes are saved beside this report.</footer></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = prepare_trades()
    trades.to_csv(OUT / "straddle_trade_metrics.csv", index=False)
    simulations = []
    summaries = []
    for gated in (False, True):
        for completed_only in (False, True):
            sim_trades, sim_summary = simulate(trades, gated, completed_only)
            simulations.append(sim_trades)
            summaries.append(sim_summary)
    simulation_trades = pd.concat(simulations, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)
    simulation_trades.to_csv(OUT / "capital_constrained_trades.csv", index=False)
    summary_all.to_csv(OUT / "strategy_summary.csv", index=False)

    quality = signal_quality(trades)
    quality.to_csv(OUT / "signal_quality.csv", index=False)
    shadow = shadow_summary(trades)
    shadow.to_csv(OUT / "one_contract_shadow_summary.csv", index=False)
    benchmark_frame = benchmark(trades)
    benchmark_frame.to_csv(OUT / "underlying_benchmark.csv", index=False)
    failures = pd.read_csv(ROOT / "selection_failures.csv", parse_dates=["entry_date", "exit_date"])
    failures.to_csv(OUT / "selection_failures.csv", index=False)
    signals = load_signals()
    signals.to_csv(OUT / "oni_signal_inputs.csv", index=False)

    hashes = {}
    for path in [ROOT / "contract_selections.csv", ROOT / "usd_hkd_daily.csv", *sorted(RAW.glob("*.json"))]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "data_vendor": "Massive",
        "test_start": "2026-04-01",
        "test_end": "2026-09-11",
        "initial_capital_hkd": INITIAL_CAPITAL_HKD,
        "tickers": list(TICKERS),
        "contract_selection": "same-strike ATM call+put, 45-70 DTE, target 55 DTE, point-in-time as_of",
        "holding_period": "20 trading sessions including entry; September truncated at latest session",
        "option_slippage_each_transaction": OPTION_SLIPPAGE_EACH_SIDE,
        "commission_usd_per_contract_per_leg_per_side": COMMISSION_USD_PER_CONTRACT_PER_LEG_PER_SIDE,
        "tax_base_case": "0% capital-gains assumption for a Hong Kong personal investor; classification not determined",
        "oni_delay_proxy_months": ONI_PUBLICATION_DELAY_MONTHS,
        "hashes": hashes,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    monthly_summary = summary_all[(summary_all["universe"] == "Monthly") & (summary_all["sample"] == "Through 2026-09-11")]
    gated_summary = summary_all[(summary_all["universe"] == "ONI-gated") & (summary_all["sample"] == "Through 2026-09-11")]
    completed_summary = summary_all[(summary_all["universe"] == "Monthly") & (summary_all["sample"] == "Completed only")]
    report = build_report(trades, monthly_summary, gated_summary, completed_summary, quality, benchmark_frame, shadow, failures)
    (ROOT / "DC_infrastructure_straddles_Apr_Sep_2026.html").write_text(report)
    print(monthly_summary.sort_values("portfolio_return", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
