#!/usr/bin/env python3
"""Point-in-time risk overlays for the ENSO cooling-equity strategy.

The module deliberately keeps policy constants fixed.  It compares each overlay with
the same investable seven-stock cooling basket and charges one-way turnover costs.
No risk-control parameter is fitted to the realized backtest outcome.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_analysis as core


ROOT = Path(__file__).resolve().parent
CALC = ROOT / "calculated"
REPORT = ROOT / "cooling_strategy_risk_management_report.html"
START_CAPITAL_HKD = 100_000.0
ANNUAL_VOL_TARGET = 0.20
MONTHLY_ES_BUDGET = 0.08
MAX_NAME_WEIGHT = 0.25
WEAK_TREND_MULTIPLIER = 0.50
DD_CAUTION = -0.10
DD_DEFENSIVE = -0.20
DD_CAUTION_MULTIPLIER = 0.50
DD_DEFENSIVE_MULTIPLIER = 0.25
HEDGE_BETA_STRONG = 0.25
HEDGE_BETA_WEAK = 0.50
CORE_WEIGHT = 0.85
SIGNAL_WEIGHT = 0.15


VARIANTS = {
    "basic": "Basic equal-weight basket",
    "inverse_vol": "Inverse-volatility core",
    "vol_target": "Inverse-vol + 20% vol target",
    "trend": "Inverse-vol + trend scaling",
    "drawdown": "Inverse-vol + drawdown brake",
    "tail_budget": "Inverse-vol + expected-shortfall budget",
    "combined": "Combined risk-managed core",
    "core_signal": "85/15 core + confirmed signal",
    "confirmed": "Confirmed ENSO signal",
    "XLI": "XLI benchmark",
}


def capped_inverse_vol(returns: pd.DataFrame, month: pd.Period) -> pd.Series:
    """Use trailing 12-month volatility and cap any stock at 25%."""
    hist = returns.loc[: month - 1, core.TICKERS].tail(12)
    vols = hist.std(ddof=1).replace(0, np.nan)
    fallback = float(np.nanmedian(vols.to_numpy()))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.10
    raw = 1 / vols.fillna(fallback).clip(lower=0.02)
    weights = raw / raw.sum()
    # Water-fill uncapped names until every weight respects the concentration cap.
    fixed = pd.Series(False, index=weights.index)
    for _ in range(len(weights)):
        over = (~fixed) & (weights > MAX_NAME_WEIGHT)
        if not over.any():
            break
        weights.loc[over] = MAX_NAME_WEIGHT
        fixed.loc[over] = True
        remaining = 1.0 - float(weights.loc[fixed].sum())
        free = ~fixed
        if free.any():
            weights.loc[free] = raw.loc[free] / raw.loc[free].sum() * remaining
    return weights / weights.sum()


def trailing_risk_controls(shadow: pd.Series, shadow_wealth: pd.Series) -> dict[str, float]:
    """Compute controls using returns known before the current test month."""
    history = shadow.dropna().tail(24)
    recent = shadow.dropna().tail(12)
    ann_vol = float(recent.std(ddof=1) * math.sqrt(12)) if len(recent) >= 6 else np.nan
    vol_multiplier = min(1.0, ANNUAL_VOL_TARGET / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0 else 1.0

    if len(history) >= 12:
        cutoff = float(history.quantile(0.10))
        es = float(history[history <= cutoff].mean())
        es_multiplier = min(1.0, MONTHLY_ES_BUDGET / abs(es)) if es < 0 else 1.0
    else:
        es, es_multiplier = np.nan, 1.0

    if len(shadow_wealth):
        shadow_dd = float(shadow_wealth.iloc[-1] / shadow_wealth.cummax().iloc[-1] - 1)
    else:
        shadow_dd = 0.0
    if shadow_dd <= DD_DEFENSIVE:
        dd_multiplier = DD_DEFENSIVE_MULTIPLIER
    elif shadow_dd <= DD_CAUTION:
        dd_multiplier = DD_CAUTION_MULTIPLIER
    else:
        dd_multiplier = 1.0
    return {
        "trailing_ann_vol": ann_vol,
        "trailing_es90": es,
        "shadow_drawdown": shadow_dd,
        "vol_multiplier": vol_multiplier,
        "es_multiplier": es_multiplier,
        "dd_multiplier": dd_multiplier,
    }


def simulate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices, _, returns, _ = core.load_data()
    original = pd.read_csv(CALC / "backtest_monthly.csv")
    original.index = pd.PeriodIndex(original["month"], freq="M")
    original_positions = pd.read_csv(CALC / "backtest_positions.csv")
    original_positions["month"] = pd.PeriodIndex(original_positions["month"], freq="M")
    original_weights = original_positions.pivot(index="month", columns="ticker", values="weight")
    months = pd.period_range(core.BACKTEST_START, core.BACKTEST_END, freq="M")

    variant_names = ["basic", "inverse_vol", "vol_target", "trend", "drawdown", "tail_budget", "combined"]
    prev_weights = {name: pd.Series(0.0, index=core.TICKERS + ["XLI"]) for name in variant_names + ["confirmed", "core_signal"]}
    shadow_returns: list[float] = []
    shadow_wealth: list[float] = []
    rows: list[dict] = []
    position_rows: list[dict] = []

    for month in months:
        eq = pd.Series(1 / len(core.TICKERS), index=core.TICKERS)
        inv = capped_inverse_vol(returns, month)
        risk = trailing_risk_controls(
            pd.Series(shadow_returns, dtype=float), pd.Series(shadow_wealth, dtype=float)
        )
        xli_prev = prices["XLI"].loc[: month - 1]
        market_on = bool(
            len(xli_prev) >= 6
            and xli_prev.iloc[-1] > xli_prev.tail(6).mean()
            and np.log(xli_prev.iloc[-1] / xli_prev.iloc[-4]) > 0
        )
        trend_multiplier = 1.0 if market_on else WEAK_TREND_MULTIPLIER

        definitions = {
            "basic": (eq, 1.0, 0.0),
            "inverse_vol": (inv, 1.0, 0.0),
            "vol_target": (inv, risk["vol_multiplier"], 0.0),
            "trend": (inv, trend_multiplier, 0.0),
            "drawdown": (inv, risk["dd_multiplier"], 0.0),
            "tail_budget": (inv, risk["es_multiplier"], 0.0),
            # The combined rule uses the most restrictive independent budget. This
            # avoids multiplying several controls and unintentionally becoming cash.
            "combined": (
                inv,
                min(risk["vol_multiplier"], risk["es_multiplier"], risk["dd_multiplier"], trend_multiplier),
                HEDGE_BETA_STRONG if market_on else HEDGE_BETA_WEAK,
            ),
        }

        month_result: dict[str, float] = {}
        full_weights: dict[str, pd.Series] = {}
        for key, (base_weights, exposure, hedge_fraction) in definitions.items():
            weights = base_weights * exposure
            beta = sum(
                float(weights[t]) * core.rolling_beta(returns[t], returns["XLI"], month)
                for t in core.TICKERS
            )
            hedge = beta * hedge_fraction
            full = pd.Series(0.0, index=core.TICKERS + ["XLI"])
            full.loc[core.TICKERS] = weights
            full.loc["XLI"] = -hedge
            turnover = float((full - prev_weights[key]).abs().sum())
            gross_return = float((weights * returns.loc[month, core.TICKERS]).sum() - hedge * returns.loc[month, "XLI"])
            cost = core.COST * turnover
            net_return = gross_return - cost
            month_result[key] = net_return
            full_weights[key] = full
            for ticker in core.TICKERS:
                position_rows.append({
                    "month": str(month), "variant": key, "ticker": ticker,
                    "weight": float(weights[ticker]), "hedge_weight": 0.0,
                    "gross_exposure": float(weights.sum()), "net_exposure": float(weights.sum() - hedge),
                    "turnover": turnover, "cost": cost,
                })
            position_rows.append({
                "month": str(month), "variant": key, "ticker": "XLI_HEDGE",
                "weight": 0.0, "hedge_weight": -hedge,
                "gross_exposure": float(weights.sum()), "net_exposure": float(weights.sum() - hedge),
                "turnover": turnover, "cost": cost,
            })
            prev_weights[key] = full

        shadow_gross = float((inv * returns.loc[month, core.TICKERS]).sum())
        shadow_returns.append(shadow_gross)
        shadow_wealth.append((shadow_wealth[-1] if shadow_wealth else 1.0) * (1 + shadow_gross))

        confirmed_full = pd.Series(0.0, index=core.TICKERS + ["XLI"])
        confirmed_full.loc[core.TICKERS] = original_weights.loc[month, core.TICKERS]
        confirmed_full.loc["XLI"] = -float(original.loc[month, "xli_hedge"])
        confirmed_turnover = float((confirmed_full - prev_weights["confirmed"]).abs().sum())
        confirmed = float(original.loc[month, "strategy"])
        for ticker in core.TICKERS + ["XLI"]:
            position_rows.append({
                "month": str(month), "variant": "confirmed", "ticker": "XLI_HEDGE" if ticker == "XLI" else ticker,
                "weight": float(confirmed_full[ticker]) if ticker != "XLI" else 0.0,
                "hedge_weight": float(confirmed_full[ticker]) if ticker == "XLI" else 0.0,
                "gross_exposure": float(confirmed_full.loc[core.TICKERS].sum()),
                "net_exposure": float(confirmed_full.sum()), "turnover": confirmed_turnover,
                "cost": core.COST * confirmed_turnover,
            })

        # Fixed strategic allocation, not a hindsight-optimized mixture. Aggregate
        # positions are netted before charging turnover, rather than averaging sleeve
        # costs that could double-count opposite trades.
        mixed_full = CORE_WEIGHT * full_weights["combined"] + SIGNAL_WEIGHT * confirmed_full
        mixed_turnover = float((mixed_full - prev_weights["core_signal"]).abs().sum())
        mixed_gross_return = float(
            (mixed_full.loc[core.TICKERS] * returns.loc[month, core.TICKERS]).sum()
            + mixed_full["XLI"] * returns.loc[month, "XLI"]
        )
        core_signal = mixed_gross_return - core.COST * mixed_turnover
        for ticker in core.TICKERS + ["XLI"]:
            position_rows.append({
                "month": str(month), "variant": "core_signal", "ticker": "XLI_HEDGE" if ticker == "XLI" else ticker,
                "weight": float(mixed_full[ticker]) if ticker != "XLI" else 0.0,
                "hedge_weight": float(mixed_full[ticker]) if ticker == "XLI" else 0.0,
                "gross_exposure": float(mixed_full.loc[core.TICKERS].sum()),
                "net_exposure": float(mixed_full.sum()), "turnover": mixed_turnover,
                "cost": core.COST * mixed_turnover,
            })
        prev_weights["confirmed"] = confirmed_full
        prev_weights["core_signal"] = mixed_full
        rows.append({
            "month": str(month), **month_result, "core_signal": core_signal,
            "confirmed": confirmed, "XLI": float(returns.loc[month, "XLI"]),
            "market_on": market_on, **risk,
            "trend_multiplier": trend_multiplier,
            "combined_exposure": definitions["combined"][1],
            "combined_hedge_fraction": definitions["combined"][2],
            "shadow_return": shadow_gross,
            "core_signal_gross": mixed_gross_return,
            "core_signal_turnover": mixed_turnover,
        })

    monthly = pd.DataFrame(rows)
    positions = pd.DataFrame(position_rows)
    summary = performance_summary(monthly, positions)
    return monthly, positions, summary


def max_drawdown_duration(wealth: pd.Series) -> int:
    underwater = wealth < wealth.cummax()
    longest = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def performance_summary(monthly: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    benchmark = monthly["XLI"].astype(float)
    basic_cagr = None
    basic_dd = None
    for key, label in VARIANTS.items():
        r = monthly[key].astype(float)
        wealth = (1 + r).cumprod()
        dd = wealth / wealth.cummax() - 1
        n = len(r)
        cagr = float(wealth.iloc[-1] ** (12 / n) - 1)
        vol = float(r.std(ddof=1) * math.sqrt(12))
        downside = r[r < 0]
        sortino = float(r.mean() / downside.std(ddof=1) * math.sqrt(12)) if len(downside) > 1 and downside.std(ddof=1) else np.nan
        max_dd = float(dd.min())
        q05 = float(r.quantile(0.05))
        es95 = float(r[r <= q05].mean())
        beta = float(r.cov(benchmark) / benchmark.var()) if benchmark.var() else np.nan
        corr = float(r.corr(benchmark))
        up = float(r[benchmark > 0].mean() / benchmark[benchmark > 0].mean()) if (benchmark > 0).any() else np.nan
        down = float(r[benchmark < 0].mean() / benchmark[benchmark < 0].mean()) if (benchmark < 0).any() else np.nan
        worst_3m = float(((1 + r).rolling(3).apply(np.prod, raw=True) - 1).min())
        if key in positions["variant"].unique():
            by_month = positions[positions["variant"] == key].drop_duplicates("month")
            turnover = float(by_month["turnover"].mean() * 12)
            annual_cost = float(by_month["cost"].mean() * 12)
            avg_gross = float(by_month["gross_exposure"].mean())
        else:
            turnover = annual_cost = np.nan
            avg_gross = float((r != 0).mean()) if key == "confirmed" else 1.0
        item = {
            "variant": key, "portfolio": label, "months": n,
            "total_return": float(wealth.iloc[-1] - 1), "cagr": cagr,
            "annual_vol": vol, "sharpe_zero_rf": float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if r.std(ddof=1) else np.nan,
            "sortino_zero_target": sortino, "max_drawdown": max_dd,
            "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
            "var95_monthly": -q05, "es95_monthly": -es95,
            "worst_month": float(r.min()), "worst_3m": worst_3m,
            "positive_month_rate": float((r > 0).mean()), "beta_xli": beta,
            "correlation_xli": corr, "up_capture": up, "down_capture": down,
            "max_drawdown_months": max_drawdown_duration(wealth),
            "annual_turnover": turnover, "annual_cost_drag": annual_cost,
            "average_gross_exposure": avg_gross,
            "ending_capital_hkd": START_CAPITAL_HKD * float(wealth.iloc[-1]),
            "max_peak_to_trough_hkd": float(((wealth.cummax() - wealth) * START_CAPITAL_HKD).max()),
        }
        rows.append(item)
        if key == "basic":
            basic_cagr, basic_dd = cagr, max_dd
    out = pd.DataFrame(rows)
    out["cagr_retention_vs_basic"] = out["cagr"] / basic_cagr
    out["drawdown_reduction_vs_basic"] = 1 - out["max_drawdown"].abs() / abs(basic_dd)
    # Predeclared feasibility gate: halve neither return nor capital protection.
    out["meets_portfolio_gate"] = (
        (out["cagr_retention_vs_basic"] >= 0.50)
        & (out["drawdown_reduction_vs_basic"] >= 0.30)
        & (out["average_gross_exposure"] <= 1.000001)
    )
    return out


def next_policy_allocation(monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | bool | str]]:
    """Policy state for September 2026, frozen using data through August 2026."""
    prices, _, returns, _ = core.load_data()
    decision_month = core.BACKTEST_END + 1
    inv = capped_inverse_vol(returns, decision_month)
    shadow = monthly["shadow_return"].astype(float)
    shadow_wealth = (1 + shadow).cumprod()
    risk = trailing_risk_controls(shadow, shadow_wealth)
    xli_prev = prices["XLI"].loc[: decision_month - 1]
    market_on = bool(
        xli_prev.iloc[-1] > xli_prev.tail(6).mean()
        and np.log(xli_prev.iloc[-1] / xli_prev.iloc[-4]) > 0
    )
    trend_multiplier = 1.0 if market_on else WEAK_TREND_MULTIPLIER
    drawdown_exposure = risk["dd_multiplier"]
    combined_exposure = min(risk["vol_multiplier"], risk["es_multiplier"], risk["dd_multiplier"], trend_multiplier)
    betas = pd.Series({t: core.rolling_beta(returns[t], returns["XLI"], decision_month) for t in core.TICKERS})
    hedge_fraction = HEDGE_BETA_STRONG if market_on else HEDGE_BETA_WEAK
    combined_hedge = float((inv * combined_exposure * betas).sum() * hedge_fraction)
    allocation = pd.DataFrame({
        "ticker": core.TICKERS,
        "inverse_vol_weight": inv.values,
        "recommended_drawdown_brake_weight": (inv * drawdown_exposure).values,
        "combined_core_weight": (inv * combined_exposure).values,
    })
    allocation = pd.concat([allocation, pd.DataFrame([{
        "ticker": "XLI hedge", "inverse_vol_weight": 0.0,
        "recommended_drawdown_brake_weight": 0.0, "combined_core_weight": -combined_hedge,
    }])], ignore_index=True)
    state: dict[str, float | bool | str] = {
        "decision_month": str(decision_month), "information_through": str(decision_month - 1),
        "market_on": market_on, "trend_multiplier": trend_multiplier,
        **risk, "recommended_exposure": drawdown_exposure,
        "combined_exposure": combined_exposure, "combined_xli_hedge": combined_hedge,
    }
    return allocation, state


def fmt_pct(value: float, digits: int = 1) -> str:
    return "n.a." if pd.isna(value) else f"{100 * float(value):.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    return "n.a." if pd.isna(value) else f"{float(value):,.{digits}f}"


def table(frame: pd.DataFrame, columns: list[tuple[str, str, object]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row[key]
            rendered = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def line_svg(monthly: pd.DataFrame, keys: list[str], drawdown: bool = False, width: int = 980, height: int = 390) -> str:
    colors = {"basic": "#b6533d", "combined": "#0b625c", "core_signal": "#1f416d", "confirmed": "#a78a2d", "XLI": "#6d7779"}
    values = {}
    for key in keys:
        wealth = (1 + monthly[key]).cumprod()
        values[key] = wealth / wealth.cummax() - 1 if drawdown else wealth * START_CAPITAL_HKD
    frame = pd.DataFrame(values)
    lo, hi = float(frame.min().min()), float(frame.max().max())
    if not drawdown:
        lo = min(lo, START_CAPITAL_HKD)
    pad = max((hi - lo) * 0.08, 1.0)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 70, 20, 38, 52
    def xy(i: int, value: float) -> tuple[float, float]:
        return left + i * (width - left - right) / (len(frame) - 1), top + (hi - value) * (height - top - bottom) / (hi - lo)
    title = "Drawdown comparison" if drawdown else "Growth of HKD 100,000"
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{title}'>"]
    for fraction in np.linspace(0, 1, 5):
        value = lo + fraction * (hi - lo); y = xy(0, value)[1]
        label = fmt_pct(value, 0) if drawdown else f"{value/1000:.0f}k"
        out.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#dce3df'/><text x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{label}</text>")
    x = left
    for key in keys:
        pts = " ".join(f"{xy(i, value)[0]:.1f},{xy(i, value)[1]:.1f}" for i, value in enumerate(frame[key]))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{colors[key]}' stroke-width='2.5'/>")
        out.append(f"<line x1='{x}' x2='{x+22}' y1='18' y2='18' stroke='{colors[key]}' stroke-width='3'/><text x='{x+27}' y='22'>{VARIANTS[key]}</text>")
        x += 220
    for i in [0, len(frame)//2, len(frame)-1]:
        out.append(f"<text x='{xy(i, lo)[0]:.1f}' y='{height-18}' text-anchor='middle'>{monthly.month.iloc[i]}</text>")
    out.append("</svg>")
    return "".join(out)


def scatter_svg(summary: pd.DataFrame, width: int = 760, height: int = 430) -> str:
    chart = summary[summary["variant"].isin(["basic", "inverse_vol", "vol_target", "trend", "drawdown", "tail_budget", "combined", "core_signal"])].copy()
    xvals = chart["max_drawdown"].abs() * 100
    yvals = chart["cagr"] * 100
    xmin, xmax = max(0.0, float(xvals.min()) - 2), float(xvals.max()) + 2
    ymin, ymax = min(0.0, float(yvals.min()) - 2), float(yvals.max()) + 3
    left, right, top, bottom = 66, 28, 26, 58
    def xy(x: float, y: float) -> tuple[float, float]:
        return left + (x-xmin)*(width-left-right)/(xmax-xmin), top + (ymax-y)*(height-top-bottom)/(ymax-ymin)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Return and drawdown tradeoff'>"]
    for value in np.linspace(xmin, xmax, 5):
        px, _ = xy(value, ymin); out.append(f"<line x1='{px:.1f}' x2='{px:.1f}' y1='{top}' y2='{height-bottom}' stroke='#e1e5e2'/><text x='{px:.1f}' y='{height-27}' text-anchor='middle'>{value:.0f}%</text>")
    for value in np.linspace(ymin, ymax, 5):
        _, py = xy(xmin, value); out.append(f"<line x1='{left}' x2='{width-right}' y1='{py:.1f}' y2='{py:.1f}' stroke='#e1e5e2'/><text x='{left-8}' y='{py+4:.1f}' text-anchor='end'>{value:.0f}%</text>")
    for _, row in chart.iterrows():
        px, py = xy(abs(row.max_drawdown)*100, row.cagr*100)
        color = "#0b625c" if row.meets_portfolio_gate else "#b6533d" if row.variant == "basic" else "#647578"
        out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='6' fill='{color}'/><text x='{px+9:.1f}' y='{py-8:.1f}'>{html.escape(row.portfolio.replace('Inverse-volatility','Inv-vol').replace('Inverse-vol','Inv-vol'))}</text>")
    out.append(f"<text x='{(left+width-right)/2:.1f}' y='{height-5}' text-anchor='middle'>Maximum drawdown (smaller is better)</text>")
    out.append(f"<text transform='translate(16 {(top+height-bottom)/2:.1f}) rotate(-90)' text-anchor='middle'>Annualized return</text></svg>")
    return "".join(out)


def stress_table(monthly: pd.DataFrame) -> pd.DataFrame:
    worst = monthly.nsmallest(8, "basic").copy()
    return worst[["month", "basic", "combined", "core_signal", "confirmed", "XLI", "combined_exposure", "market_on"]]


def report_html(monthly: pd.DataFrame, summary: pd.DataFrame, allocation: pd.DataFrame, state: dict) -> str:
    basic = summary.set_index("variant").loc["basic"]
    combined = summary.set_index("variant").loc["combined"]
    mixed = summary.set_index("variant").loc["core_signal"]
    feasible = summary[summary["meets_portfolio_gate"]].sort_values("cagr", ascending=False)
    recommended = feasible.iloc[0] if len(feasible) else summary.loc[summary["max_drawdown"].idxmax()]
    ranking = summary[summary["variant"] != "XLI"].sort_values("cagr", ascending=False)
    summary_table = table(ranking, [
        ("portfolio", "Portfolio", None), ("cagr", "CAGR", fmt_pct), ("annual_vol", "Vol", fmt_pct),
        ("max_drawdown", "Max DD", fmt_pct), ("es95_monthly", "Monthly ES 95", fmt_pct),
        ("sharpe_zero_rf", "Sharpe", fmt_num), ("sortino_zero_target", "Sortino", fmt_num),
        ("cagr_retention_vs_basic", "CAGR retained", fmt_pct),
        ("drawdown_reduction_vs_basic", "DD reduced", fmt_pct),
        ("average_gross_exposure", "Avg gross", fmt_pct),
        ("meets_portfolio_gate", "Passes gate", lambda x: "Yes" if x else "No"),
    ])
    capital_table = table(ranking, [
        ("portfolio", "Portfolio", None), ("ending_capital_hkd", "Ending HKD", lambda x: f"HK${x:,.0f}"),
        ("max_peak_to_trough_hkd", "Max peak-to-trough / initial", lambda x: f"HK${x:,.0f}"),
        ("worst_month", "Worst month", fmt_pct), ("worst_3m", "Worst 3 months", fmt_pct),
        ("annual_turnover", "Turnover/year", fmt_pct), ("annual_cost_drag", "Cost/year", fmt_pct),
        ("beta_xli", "XLI beta", fmt_num), ("down_capture", "Down capture", fmt_pct),
    ])
    stress = stress_table(monthly)
    stress_html = table(stress, [
        ("month", "Basic basket stress month", None), ("basic", "Basic", fmt_pct),
        ("combined", "Combined core", fmt_pct), ("core_signal", "85/15", fmt_pct),
        ("confirmed", "ENSO signal", fmt_pct), ("XLI", "XLI", fmt_pct),
        ("combined_exposure", "Core gross", fmt_pct), ("market_on", "Trend positive", lambda x: "Yes" if x else "No"),
    ])
    allocation_html = table(allocation, [
        ("ticker", "Ticker", None), ("inverse_vol_weight", "Unscaled inverse-vol", fmt_pct),
        ("recommended_drawdown_brake_weight", "Recommended weight", fmt_pct),
        ("combined_core_weight", "Combined-core weight", fmt_pct),
    ])
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Risk management for the ENSO cooling-equity strategy</title><style>
:root{{--ink:#172528;--muted:#617174;--paper:#f3f0e7;--card:#fffefa;--navy:#1f416d;--teal:#0b625c;--rust:#b6533d;--line:#d6ddda}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.56 Inter,system-ui,-apple-system,sans-serif}}header{{padding:58px max(5vw,28px);background:linear-gradient(135deg,#102f3a,#0c5b55);color:white}}h1{{font:500 clamp(38px,6vw,66px)/1.04 Georgia,serif;margin:8px 0;max-width:1100px}}.deck{{font-size:19px;max-width:920px;color:#d7e8e4}}.meta{{display:flex;gap:20px;flex-wrap:wrap;color:#b8d4cf;font-size:13px;margin-top:24px}}main{{max-width:1200px;margin:auto;padding:40px 24px 80px}}h2{{font:500 34px Georgia,serif;margin:52px 0 12px}}h3{{font-size:20px;margin:26px 0 8px}}p{{max-width:950px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.kpi,.card{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:18px;box-shadow:0 5px 18px rgba(22,52,48,.05)}}.kpi b{{display:block;color:var(--teal);font-size:28px}}.kpi span,.small{{font-size:13px;color:var(--muted)}}.callout{{border-left:5px solid var(--rust);background:#fff7ed;border-radius:0 9px 9px 0;padding:17px 20px;margin:18px 0;max-width:1030px}}.good{{border-left-color:var(--teal);background:#edf7f3}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:9px;margin:14px 0}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{padding:10px 12px;border-bottom:1px solid #e3e8e5;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#e8f1ee;color:#35575a;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}.chart{{background:white;border:1px solid var(--line);border-radius:11px;padding:12px;overflow:auto}}svg{{min-width:700px}}svg text{{font:12px system-ui;fill:#526669}}.formula{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#edf1ee;border-radius:7px;padding:13px;overflow:auto}}li{{margin:7px 0}}a{{color:#086b65}}footer{{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}@media(max-width:850px){{.kpis,.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><div class='small' style='color:#b8d4cf;text-transform:uppercase;letter-spacing:.13em'>Point-in-time portfolio controls</div><h1>Risk management for the ENSO cooling-equity strategy</h1><p class='deck'>Seven liquid cooling-infrastructure equities, tested as a basic basket and with independent volatility, trend, drawdown, tail-loss and beta controls.</p><div class='meta'><span>Jun 2021–Aug 2026</span><span>63 monthly observations</span><span>HK$100,000 initial capital</span><span>15 bp one-way turnover cost</span><span>No leverage</span></div></header><main>
<section><h2>Decision first</h2><div class='kpis'><div class='kpi'><b>{fmt_pct(recommended.cagr)}</b><span>recommended CAGR</span></div><div class='kpi'><b>{fmt_pct(recommended.max_drawdown)}</b><span>recommended maximum drawdown</span></div><div class='kpi'><b>{fmt_pct(recommended.drawdown_reduction_vs_basic)}</b><span>drawdown reduction vs basic</span></div><div class='kpi'><b>{fmt_pct(recommended.cagr_retention_vs_basic)}</b><span>basic-basket CAGR retained</span></div></div>
<div class='callout good'><strong>Recommended implementation: {html.escape(recommended.portfolio)}.</strong> It is the highest-CAGR test that passed the predeclared portfolio gate: retain at least 50% of the basic basket's CAGR, reduce maximum drawdown by at least 30%, and use no leverage. Starting from HK$100,000, the historical ending value was HK${recommended.ending_capital_hkd:,.0f}. This is a selection rule applied after comparing the fixed policies; it is not proof of future superiority.</div>
<p>The basic basket delivered {fmt_pct(basic.cagr)} annualized but lost as much as {fmt_pct(basic.max_drawdown)} from a prior peak. The combined core delivered {fmt_pct(combined.cagr)} with a {fmt_pct(combined.max_drawdown)} drawdown. The fixed 85/15 mix with the sparse ENSO signal delivered {fmt_pct(mixed.cagr)} with a {fmt_pct(mixed.max_drawdown)} drawdown. The risk controls reduce equity exposure in adverse states; they do not eliminate loss.</p>
<div class='callout'><strong>Interpretation constraint.</strong> This period overlaps the AI data-center re-rating, contains only one completed strong El Niño, and uses a survivor-selected cooling universe. Return retention is therefore a historical diagnostic, not an expected return. Monthly bars cannot validate intramonth stops, option hedges, limit orders or gap risk.</div>{summary_table}</section>
<section><h2>Last executable allocation state</h2><p>The last complete-month decision is for {state['decision_month']}, using information only through {state['information_through']}. It is shown for audit and must not be retroactively entered after observing part of September. The next clean rebalance can be frozen only after September closes.</p><div class='kpis'><div class='kpi'><b>{fmt_pct(state['shadow_drawdown'])}</b><span>shadow-core drawdown</span></div><div class='kpi'><b>{fmt_pct(state['recommended_exposure'])}</b><span>recommended gross-long exposure</span></div><div class='kpi'><b>{fmt_pct(state['combined_exposure'])}</b><span>combined-core gross-long exposure</span></div><div class='kpi'><b>{'Yes' if state['market_on'] else 'No'}</b><span>prior XLI trend positive</span></div></div>{allocation_html}<p>The report recommends the drawdown-braked column because it alone passes the return-retention gate. The combined column is included for a stricter loss budget; its XLI hedge is {fmt_pct(-float(state['combined_xli_hedge']))} of capital. Cash is the difference between 100% and gross-long exposure.</p></section>
<section><h2>Capital and tail-risk view</h2>{capital_table}<div class='grid'><div class='chart'>{line_svg(monthly,["basic","combined","core_signal","XLI"])}</div><div class='chart'>{line_svg(monthly,["basic","combined","core_signal","XLI"],True)}</div></div></section>
<section><h2>Return retained versus drawdown</h2><p>Points higher and farther left are preferable. Green points pass the predeclared portfolio gate; the red point is the unmanaged reference.</p><div class='chart'>{scatter_svg(summary)}</div></section>
<section><h2>What each control does</h2><div class='grid'>
<div class='card'><h3>Concentration cap + inverse volatility</h3><p>Weights are proportional to the inverse of each stock's trailing 12-month volatility, then capped at 25%. This limits a volatile stock's risk contribution and prevents one name from exceeding one quarter of long capital.</p></div>
<div class='card'><h3>20% volatility target</h3><p>Gross exposure is the smaller of 100% and 20% divided by the annualized volatility of the prior 12 shadow-core returns. It de-risks after observed volatility rises and never adds leverage after calm periods.</p></div>
<div class='card'><h3>XLI trend rule</h3><p>Exposure is 100% when prior-month XLI is above its trailing six-month mean and its trailing three-month return is positive; otherwise it is 50%. All inputs precede the test month.</p></div>
<div class='card'><h3>Drawdown brake</h3><p>The unscaled inverse-volatility shadow portfolio determines the state: 100% above −10%, 50% from −10% to −20%, and 25% below −20%. A shadow portfolio permits automatic re-entry when the underlying basket recovers.</p></div>
<div class='card'><h3>Expected-shortfall budget</h3><p>The mean return in the worst 10% of the prior 24 months estimates tail loss. Exposure is capped so this estimate does not exceed 8% in one month. The 10% estimator is used for sizing; the report separately shows 95% realized expected shortfall.</p></div>
<div class='card'><h3>Regime-dependent XLI hedge</h3><p>The combined core shorts 25% of estimated stock beta in a positive XLI regime and 50% in a weak regime. Betas use up to 24 prior months and are clipped to 0.25–2.00 per stock to avoid unstable hedge sizes.</p></div>
</div><h3>Combination rule</h3><p>The combined core uses the <em>minimum</em> of the volatility, expected-shortfall, drawdown and trend multipliers. It does not multiply them, because multiplication can unintentionally drive a diversified portfolio close to cash. The core–signal portfolio places 85% in that combined core and 15% in the existing confirmed ENSO sleeve.</p>
<div class='formula'>wᵢ,t = exposureₜ × capped[(1/σᵢ,t−1) / Σⱼ(1/σⱼ,t−1)]<br>exposureₜ = min(1, 20%/σportfolio,t−1, 8%/|ES90,t−1|, drawdown brake, trend multiplier)<br>Rnet,t = Σᵢwᵢ,tRᵢ,t − hedgeₜRXLI,t − 0.0015×turnoverₜ</div></section>
<section><h2>Stress-month evidence</h2><p>These are the eight worst months of the basic basket, selected only for diagnosis. They show when each policy helped or failed; they were not used to set thresholds.</p>{stress_html}</section>
<section><h2>Metric definitions</h2><ul>
<li><strong>Total return:</strong> Π(1 + rₜ) − 1. <strong>CAGR:</strong> [Π(1 + rₜ)]<sup>12/N</sup> − 1.</li>
<li><strong>Annualized volatility:</strong> sample standard deviation of monthly returns × √12. <strong>Sharpe:</strong> mean monthly return / monthly standard deviation × √12. A zero cash rate is used.</li>
<li><strong>Sortino:</strong> mean monthly return / standard deviation of negative monthly returns × √12. The target return is zero.</li>
<li><strong>Maximum drawdown:</strong> minimum of wealthₜ / prior peak wealthₜ − 1. <strong>Calmar:</strong> CAGR / absolute maximum drawdown.</li>
<li><strong>Monthly 95% VaR:</strong> the positive magnitude of the empirical 5th-percentile monthly return. <strong>Monthly 95% expected shortfall:</strong> the positive magnitude of the mean return at or below that percentile. Both are historical and based on only 63 months.</li>
<li><strong>XLI beta:</strong> Cov(rportfolio, rXLI) / Var(rXLI). <strong>Up/down capture:</strong> mean portfolio return divided by mean XLI return in XLI-positive/negative months.</li>
<li><strong>CAGR retained:</strong> portfolio CAGR / basic-basket CAGR. <strong>Drawdown reduced:</strong> 1 − |portfolio max drawdown| / |basic max drawdown|.</li>
<li><strong>Turnover:</strong> sum of absolute monthly changes in all long and hedge weights, annualized. <strong>Cost drag:</strong> turnover × 15 bp, annualized; it excludes taxes, bid–ask slippage and financing.</li>
</ul></section>
<section><h2>Implementation and limitations</h2><ol>
<li>Rebalance monthly after the information date. Calculate every trailing statistic through the previous month.</li>
<li>Use adjusted Massive bars and verify splits/dividends. Do not substitute unadjusted prices.</li>
<li>Keep a 25% name limit, 100% gross-long limit and the fixed 85/15 strategic allocation. Do not raise risk to recover losses.</li>
<li>Monitor model/data failure separately: a missing price, stale ONI release, or unavailable news feed should block a new signal, not be imputed as bullish.</li>
<li>Tax is excluded because Hong Kong tax treatment depends on facts, residence and whether activity is considered a trade. Confirm with a tax professional. Borrow costs and short-sale constraints on XLI are also excluded.</li>
</ol><p>Detailed code architecture, rationale, data provenance, reproduction commands and known biases are documented in <a href='README.md'>README.md</a>. Auditable monthly returns are in <a href='calculated/risk_management_monthly.csv'>risk_management_monthly.csv</a>; portfolio metrics are in <a href='calculated/risk_management_summary.csv'>risk_management_summary.csv</a>; weights and costs are in <a href='calculated/risk_management_positions.csv'>risk_management_positions.csv</a>.</p></section>
<footer>Research simulation, not investment, tax or legal advice. Historical loss controls can fail during gaps, structural breaks, illiquidity or correlated sell-offs.</footer></main></body></html>"""


def main() -> None:
    CALC.mkdir(parents=True, exist_ok=True)
    monthly, positions, summary = simulate()
    allocation, state = next_policy_allocation(monthly)
    monthly.to_csv(CALC / "risk_management_monthly.csv", index=False)
    positions.to_csv(CALC / "risk_management_positions.csv", index=False)
    summary.to_csv(CALC / "risk_management_summary.csv", index=False)
    allocation.to_csv(CALC / "risk_management_next_allocation.csv", index=False)
    policy = {
        "initial_capital_hkd": START_CAPITAL_HKD,
        "annual_vol_target": ANNUAL_VOL_TARGET,
        "monthly_es_budget": MONTHLY_ES_BUDGET,
        "max_name_weight": MAX_NAME_WEIGHT,
        "weak_trend_multiplier": WEAK_TREND_MULTIPLIER,
        "drawdown_thresholds": [DD_CAUTION, DD_DEFENSIVE],
        "drawdown_multipliers": [1.0, DD_CAUTION_MULTIPLIER, DD_DEFENSIVE_MULTIPLIER],
        "xli_hedge_beta_fraction": {"positive_regime": HEDGE_BETA_STRONG, "weak_regime": HEDGE_BETA_WEAK},
        "core_signal_mix": {"risk_managed_core": CORE_WEIGHT, "confirmed_enso_signal": SIGNAL_WEIGHT},
        "transaction_cost_one_way": core.COST,
        "selection_gate": {"minimum_cagr_retention": 0.50, "minimum_drawdown_reduction": 0.30, "maximum_gross": 1.0},
    }
    policy["last_complete_month_state"] = state
    (CALC / "risk_policy.json").write_text(json.dumps(policy, indent=2))
    REPORT.write_text(report_html(monthly, summary, allocation, state))
    print(summary[["portfolio", "cagr", "annual_vol", "max_drawdown", "es95_monthly", "cagr_retention_vs_basic", "drawdown_reduction_vs_basic", "meets_portfolio_gate"]].to_string(index=False))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
