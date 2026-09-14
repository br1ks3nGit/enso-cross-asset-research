#!/usr/bin/env python3
"""Walk-forward COP/XLE strategies triggered by delayed strong El Nino signals."""

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
HORIZONS = (1, 2, 5, 10, 20, 40, 60)
TEST_START = pd.Timestamp("2020-01-01")
COMMISSION_PER_ORDER = 0.001  # 10 bp on each order's notional
PAIR_ROUND_TRIP_COST = 2 * COMMISSION_PER_ORDER  # 50/50 legs, entry plus exit
SHORT_TERM_TAX = 0.37
LONG_TERM_TAX = 0.20
INITIAL_CAPITAL = 100_000.0
MIN_TRAIN = 3

RULES = {
    "S1 Strong level": lambda r: r.ONI >= 1.0,
    "S2 Very strong": lambda r: r.ONI >= 1.5,
    "S3 Strong and rising": lambda r: r.ONI >= 1.0 and r.dONI > 0,
    "S4 Rapid strengthening": lambda r: r.ONI >= 1.0 and r.dONI >= 0.3,
    "S5 Strong but cooling": lambda r: r.ONI >= 1.0 and r.dONI < 0,
}


def load_asset(ticker: str) -> pd.DataFrame:
    payload = json.loads((RAW / f"massive_{ticker}_daily.json").read_text())
    rows = payload["results"]
    frame = pd.DataFrame({
        "date": [pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert(None).normalize() for r in rows],
        "open": [float(r["o"]) for r in rows],
        "close": [float(r["c"]) for r in rows],
    }).set_index("date").sort_index()
    divs = json.loads((RAW / f"massive_{ticker}_dividends.json").read_text()).get("results", [])
    div = pd.Series(0.0, index=frame.index)
    for record in divs:
        date = pd.Timestamp(record["ex_dividend_date"])
        amount = record.get("split_adjusted_cash_amount", record.get("cash_amount", 0.0))
        if date in div.index and amount is not None:
            div.loc[date] += float(amount)
    frame["dividend"] = div
    frame["total_return"] = (frame["close"] + frame["dividend"]) / frame["close"].shift(1) - 1
    frame["total_index"] = (1 + frame["total_return"].fillna(0)).cumprod()
    return frame


def holding_return(frame: pd.DataFrame, entry: pd.Timestamp, horizon: int) -> tuple[float, pd.Timestamp] | tuple[float, pd.NaT]:
    dates = frame.index
    pos = dates.searchsorted(entry)
    if pos >= len(dates) or dates[pos] != entry:
        return math.nan, pd.NaT
    exit_pos = pos + horizon - 1
    if exit_pos >= len(dates):
        return math.nan, pd.NaT
    exit_date = dates[exit_pos]
    # Entry is at the first session's open. A dividend on entry date belongs to the prior holder.
    wealth = frame.loc[entry, "close"] / frame.loc[entry, "open"]
    if exit_pos > pos:
        wealth *= float((1 + frame.iloc[pos + 1:exit_pos + 1]["total_return"]).prod())
    return float(wealth - 1), exit_date


def first_session_after(dates: pd.DatetimeIndex, timestamp: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    if timestamp < dates.min():
        return pd.NaT
    pos = dates.searchsorted(timestamp, side="right")
    return dates[pos] if pos < len(dates) else pd.NaT


def load_signals(common_dates: pd.DatetimeIndex) -> pd.DataFrame:
    oni = pd.read_csv(ONI_FILE)
    oni["centered_month"] = pd.to_datetime(oni["Date"]).dt.to_period("M")
    oni = oni.sort_values("centered_month")
    oni["dONI"] = oni["ONI"].diff()
    oni["information_month"] = oni["centered_month"] + 2
    oni["information_cutoff"] = oni["information_month"].dt.to_timestamp(how="end").dt.normalize()
    oni["entry_date"] = oni["information_cutoff"].map(lambda x: first_session_after(common_dates, x))
    oni["strength"] = pd.cut(
        oni["ONI"], bins=[-np.inf, 1.0, 1.5, 2.0, np.inf], right=False,
        labels=["Below strong", "Strong 1.0–1.4", "Very strong 1.5–1.9", "Extreme ≥2.0"],
    ).astype(str)
    return oni


def ttest(values: pd.Series) -> tuple[float, float]:
    x = values.dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return math.nan, math.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


def corr_test(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return math.nan, math.nan
    r, p = stats.pearsonr(z.iloc[:, 0], z.iloc[:, 1])
    return float(r), float(p)


def classify_regime(xle: pd.DataFrame, entry: pd.Timestamp) -> tuple[str, float, float]:
    prior = xle.loc[:entry].iloc[:-1]
    if len(prior) < 127:
        return "Unavailable", math.nan, math.nan
    trend = float(prior["total_index"].iloc[-1] / prior["total_index"].iloc[-127] - 1)
    logret = np.log1p(prior["total_return"].dropna())
    vol20 = logret.rolling(20).std(ddof=1) * math.sqrt(252)
    current_vol = float(vol20.iloc[-1])
    expanding_median = float(vol20.dropna().median())
    trend_label = "Energy uptrend" if trend > 0 else "Energy downtrend"
    vol_label = "high volatility" if current_vol > expanding_median else "low volatility"
    return f"{trend_label}, {vol_label}", trend, current_vol


def annual_tax_equity(trades: pd.DataFrame, tax_rate: float = SHORT_TERM_TAX) -> tuple[float, float, float, list[dict]]:
    equity = INITIAL_CAPITAL
    tax_total = 0.0
    commission_total = 0.0
    curve = [{"date": str(TEST_START.date()), "equity": equity, "event": "start"}]
    for year, group in trades.sort_values("exit_date").groupby(trades["exit_date"].dt.year, sort=True):
        year_profit = 0.0
        for _, trade in group.iterrows():
            commission_total += equity * PAIR_ROUND_TRIP_COST
            profit = equity * float(trade["net_pre_tax_return"])
            equity += profit
            year_profit += profit
            curve.append({"date": str(trade["exit_date"].date()), "equity": equity, "event": "trade exit"})
        tax = tax_rate * max(year_profit, 0.0)
        equity -= tax
        tax_total += tax
        curve.append({"date": f"{int(year)}-12-31", "equity": equity, "event": "tax settlement"})
    return equity, tax_total, commission_total, curve


def max_drawdown(curve: list[dict]) -> float:
    values = pd.Series([row["equity"] for row in curve], dtype=float)
    return float((values / values.cummax() - 1).min())


def equity_svg(curve: pd.DataFrame) -> str:
    width, height, left, top, right, bottom = 1040, 320, 70, 35, 25, 45
    dates = pd.to_datetime(curve["date"])
    values = curve["equity"].astype(float) / INITIAL_CAPITAL - 1
    xmin, xmax = dates.min().value, dates.max().value
    ymin, ymax = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    if ymin == ymax:
        ymin, ymax = ymin - 0.01, ymax + 0.01
    x = left + (dates.astype("int64") - xmin) / max(xmax - xmin, 1) * (width - left - right)
    y = top + (ymax - values) / (ymax - ymin) * (height - top - bottom)
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    zero_y = top + ymax / (ymax - ymin) * (height - top - bottom)
    labels = "".join(
        f"<text x='{left + i * (width-left-right)/3:.0f}' y='{height-15}' fill='#9fb0c4' font-size='12' text-anchor='middle'>{year}</text>"
        for i, year in enumerate([2020, 2022, 2024, 2026])
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Best strategy after-tax equity curve'>"
        f"<rect width='100%' height='100%' fill='#0f1d2d'/><line x1='{left}' x2='{width-right}' y1='{zero_y:.1f}' y2='{zero_y:.1f}' stroke='#60738a'/>"
        f"<polyline points='{points}' fill='none' stroke='#38bdf8' stroke-width='3'/>{labels}"
        f"<text x='15' y='{top+8}' fill='#9fb0c4' font-size='12'>{ymax:.1%}</text><text x='15' y='{height-bottom}' fill='#9fb0c4' font-size='12'>{ymin:.1%}</text></svg>"
    )


def heatmap_table(pivot: pd.DataFrame) -> str:
    limit = max(abs(float(np.nanmin(pivot.values))), abs(float(np.nanmax(pivot.values))), 0.01)
    head = "<th>Signal rule</th>" + "".join(f"<th>{int(h)}d</th>" for h in pivot.columns)
    rows = []
    for label, row in pivot.iterrows():
        cells = [f"<td>{html.escape(label)}</td>"]
        for value in row:
            if pd.isna(value):
                cells.append("<td style='color:#8fa1b5'>n.a.</td>")
                continue
            intensity = min(abs(float(value)) / limit, 1.0)
            color = f"rgba({52 if value >= 0 else 251},{211 if value >= 0 else 113},{153 if value >= 0 else 133},{0.14 + .48*intensity:.2f})"
            cells.append(f"<td style='background:{color};font-weight:700'>{value:.1%}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table class='heat'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def table(frame: pd.DataFrame, columns: list[tuple[str, str, callable]]) -> str:
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cop, xle = load_asset("COP"), load_asset("XLE")
    common_dates = cop.index.intersection(xle.index).sort_values()
    cop, xle = cop.reindex(common_dates), xle.reindex(common_dates)
    last_date = common_dates.max()
    signals = load_signals(common_dates)

    observation_rows = []
    for rule_name, predicate in RULES.items():
        eligible = signals[signals.apply(predicate, axis=1)].copy()
        for horizon in HORIZONS:
            history = []
            for _, signal in eligible.iterrows():
                entry = signal["entry_date"]
                if pd.isna(entry):
                    continue
                cop_ret, cop_exit = holding_return(cop, entry, horizon)
                xle_ret, xle_exit = holding_return(xle, entry, horizon)
                exit_date = cop_exit if cop_exit == xle_exit else pd.NaT
                spread = cop_ret - xle_ret if not pd.isna(exit_date) else math.nan
                prior = [r for r in history if not pd.isna(r["exit_date"]) and r["exit_date"] < entry and not pd.isna(r["spread_return"])]
                train_n = len(prior)
                train_mean = float(np.mean([r["spread_return"] for r in prior])) if prior else math.nan
                direction = float(np.sign(train_mean)) if train_n >= MIN_TRAIN and train_mean != 0 else math.nan
                regime, trend126, vol20 = classify_regime(xle, entry)
                row = {
                    "rule": rule_name,
                    "centered_month": str(signal["centered_month"]),
                    "information_month": str(signal["information_month"]),
                    "entry_date": entry,
                    "exit_date": exit_date,
                    "horizon_days": horizon,
                    "ONI": float(signal["ONI"]),
                    "dONI": float(signal["dONI"]),
                    "strength": signal["strength"],
                    "train_n": train_n,
                    "train_mean_spread": train_mean,
                    "direction": direction,
                    "direction_label": "Long COP / short XLE" if direction == 1 else ("Long XLE / short COP" if direction == -1 else "Insufficient training"),
                    "cop_return": cop_ret,
                    "xle_return": xle_ret,
                    "spread_return": spread,
                    "model_gross_return": 0.5 * direction * spread if not pd.isna(direction) and not pd.isna(spread) else math.nan,
                    "regime": regime,
                    "xle_trailing_126d_return": trend126,
                    "xle_volatility_20d": vol20,
                    "is_test": bool(entry >= TEST_START),
                }
                history.append(row)
                observation_rows.append(row)
    observations = pd.DataFrame(observation_rows)
    observations.to_csv(OUT / "walkforward_signal_observations.csv", index=False)

    test_obs = observations[(observations["is_test"]) & observations["direction"].notna()].copy()
    quality_rows = []
    for (rule, horizon), group in test_obs.groupby(["rule", "horizon_days"], sort=False):
        complete = group.dropna(subset=["model_gross_return"])
        r, rp = corr_test(complete["ONI"], complete["spread_return"])
        t, p = ttest(complete["model_gross_return"])
        quality_rows.append({
            "rule": rule, "horizon_days": horizon,
            "signals_total": len(group), "signals_complete": len(complete),
            "unique_episodes": int((complete["centered_month"].str[:4].astype(int).diff().abs() > 1).sum() + (len(complete) > 0)),
            "mean_spread_return": complete["spread_return"].mean(),
            "mean_model_gross_return": complete["model_gross_return"].mean(),
            "directional_hit_rate": (complete["model_gross_return"] > 0).mean() if len(complete) else math.nan,
            "signal_return_r": r, "signal_return_p": rp,
            "mean_t": t, "mean_p": p,
        })
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT / "signal_quality.csv", index=False)

    trade_rows = []
    for (rule, horizon), group in test_obs.groupby(["rule", "horizon_days"], sort=False):
        next_free = pd.Timestamp.min
        for _, obs in group.sort_values("entry_date").iterrows():
            if pd.isna(obs["exit_date"]) or obs["entry_date"] <= next_free:
                continue
            gross = float(obs["model_gross_return"])
            trade_rows.append({
                **obs.to_dict(),
                "gross_return": gross,
                "commission_return": PAIR_ROUND_TRIP_COST,
                "net_pre_tax_return": gross - PAIR_ROUND_TRIP_COST,
            })
            next_free = obs["exit_date"]
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(OUT / "non_overlapping_trades.csv", index=False)

    summary_rows, curves = [], {}
    for rule in RULES:
        for horizon in HORIZONS:
            group = trades[(trades["rule"] == rule) & (trades["horizon_days"] == horizon)].copy()
            if group.empty:
                summary_rows.append({
                    "rule": rule, "horizon_days": horizon, "trades": 0,
                    "gross_cumulative_return": math.nan, "net_pre_tax_cumulative_return": math.nan,
                    "after_tax_cumulative_return": math.nan, "mean_net_trade_return": math.nan,
                    "median_net_trade_return": math.nan, "win_rate": math.nan,
                    "commissions_paid": 0.0, "tax_paid": 0.0, "max_closed_trade_drawdown": math.nan,
                })
                continue
            final_equity, tax_paid, commissions_paid, curve = annual_tax_equity(group)
            curves[(rule, horizon)] = curve
            gross_cum = float((1 + group["gross_return"]).prod() - 1)
            pretax_cum = float((1 + group["net_pre_tax_return"]).prod() - 1)
            summary_rows.append({
                "rule": rule, "horizon_days": horizon, "trades": len(group),
                "gross_cumulative_return": gross_cum,
                "net_pre_tax_cumulative_return": pretax_cum,
                "after_tax_cumulative_return": final_equity / INITIAL_CAPITAL - 1,
                "mean_net_trade_return": group["net_pre_tax_return"].mean(),
                "median_net_trade_return": group["net_pre_tax_return"].median(),
                "win_rate": (group["net_pre_tax_return"] > 0).mean(),
                "commissions_paid": commissions_paid,
                "tax_paid": tax_paid,
                "max_closed_trade_drawdown": max_drawdown(curve),
            })
    strategy_summary = pd.DataFrame(summary_rows).sort_values(["after_tax_cumulative_return", "trades"], ascending=[False, False])
    strategy_summary.to_csv(OUT / "strategy_summary.csv", index=False)

    strength_rows = []
    base = test_obs[test_obs["rule"] == "S1 Strong level"].dropna(subset=["model_gross_return"])
    for (strength, horizon), group in base.groupby(["strength", "horizon_days"], observed=True):
        t, p = ttest(group["model_gross_return"])
        strength_rows.append({
            "strength": strength, "horizon_days": horizon, "n": len(group),
            "mean_spread_return": group["spread_return"].mean(),
            "mean_model_gross_return": group["model_gross_return"].mean(),
            "directional_hit_rate": (group["model_gross_return"] > 0).mean(),
            "p": p,
        })
    strength = pd.DataFrame(strength_rows)
    strength.to_csv(OUT / "signal_strength_results.csv", index=False)

    regime_rows = []
    for (rule, horizon, regime), group in trades.groupby(["rule", "horizon_days", "regime"], sort=False):
        regime_rows.append({
            "rule": rule, "horizon_days": horizon, "regime": regime, "trades": len(group),
            "mean_net_trade_return": group["net_pre_tax_return"].mean(),
            "win_rate": (group["net_pre_tax_return"] > 0).mean(),
            "cumulative_net_pre_tax_return": float((1 + group["net_pre_tax_return"]).prod() - 1),
        })
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(OUT / "market_regime_results.csv", index=False)

    # Passive energy benchmark over the full test interval, including dividends.
    entry_date = common_dates[common_dates.searchsorted(TEST_START)]
    xle_full_ret, benchmark_exit = holding_return(xle, entry_date, len(common_dates[common_dates >= entry_date]))
    benchmark_pre_tax = xle_full_ret - 2 * COMMISSION_PER_ORDER
    benchmark_tax = LONG_TERM_TAX * max(benchmark_pre_tax, 0.0) * INITIAL_CAPITAL
    benchmark_after_tax = benchmark_pre_tax - benchmark_tax / INITIAL_CAPITAL
    benchmark_rows = [{
        "benchmark": "XLE buy and hold", "entry_date": entry_date, "exit_date": benchmark_exit,
        "gross_return": xle_full_ret, "net_pre_tax_return": benchmark_pre_tax,
        "tax_rate": LONG_TERM_TAX, "tax_paid": benchmark_tax,
        "after_tax_return": benchmark_after_tax,
    }]

    best = strategy_summary.iloc[0]
    best_key = (best["rule"], int(best["horizon_days"]))
    best_trades = trades[(trades["rule"] == best_key[0]) & (trades["horizon_days"] == best_key[1])].copy()
    best_regimes = regimes[(regimes["rule"] == best_key[0]) & (regimes["horizon_days"] == best_key[1])].sort_values("mean_net_trade_return", ascending=False)
    matched = best_trades[["entry_date", "exit_date", "xle_return"]].copy()
    matched["gross_return"] = matched["xle_return"]
    matched["net_pre_tax_return"] = matched["gross_return"] - 2 * COMMISSION_PER_ORDER
    matched_final, matched_tax, _, _ = annual_tax_equity(matched, SHORT_TERM_TAX)
    benchmark_rows.append({
        "benchmark": "XLE during best-strategy windows", "entry_date": matched["entry_date"].min(),
        "exit_date": matched["exit_date"].max(),
        "gross_return": float((1 + matched["gross_return"]).prod() - 1),
        "net_pre_tax_return": float((1 + matched["net_pre_tax_return"]).prod() - 1),
        "tax_rate": SHORT_TERM_TAX, "tax_paid": matched_tax,
        "after_tax_return": matched_final / INITIAL_CAPITAL - 1,
    })
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(OUT / "energy_benchmark.csv", index=False)

    sensitivity_rows = []
    for rate in (0.00, 0.24, 0.37):
        final, tax_paid, _, _ = annual_tax_equity(best_trades, rate)
        sensitivity_rows.append({"case": "Best COP/XLE strategy", "tax_rate": rate, "after_tax_return": final / INITIAL_CAPITAL - 1, "tax_paid": tax_paid})
    for rate in (0.00, 0.15, 0.20):
        tax = rate * max(benchmark_pre_tax, 0.0) * INITIAL_CAPITAL
        sensitivity_rows.append({"case": "XLE buy and hold", "tax_rate": rate, "after_tax_return": benchmark_pre_tax - tax / INITIAL_CAPITAL, "tax_paid": tax})
    tax_sensitivity = pd.DataFrame(sensitivity_rows)
    tax_sensitivity.to_csv(OUT / "tax_sensitivity.csv", index=False)

    # Self-contained visualizations.
    pivot = strategy_summary.pivot(index="rule", columns="horizon_days", values="after_tax_cumulative_return").reindex(index=RULES)
    curve_frame = pd.DataFrame(curves[best_key])
    curve_frame["date"] = pd.to_datetime(curve_frame["date"])
    equity_chart = equity_svg(curve_frame)
    heatmap = heatmap_table(pivot)

    hashes = {}
    for path in [ONI_FILE, RAW / "massive_COP_daily.json", RAW / "massive_XLE_daily.json", RAW / "massive_COP_dividends.json", RAW / "massive_XLE_dividends.json"]:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "generated": str(pd.Timestamp.now(tz="Asia/Hong_Kong")),
        "last_common_price_date": str(last_date.date()),
        "test_start": str(TEST_START.date()),
        "massive_endpoint": "/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}",
        "massive_dividend_endpoint": "/stocks/v1/dividends",
        "adjusted": "split-adjusted; dividends added from Massive reference dividends",
        "oni_delay_months": 2,
        "commission_per_order_notional": COMMISSION_PER_ORDER,
        "pair_round_trip_cost_of_capital": PAIR_ROUND_TRIP_COST,
        "short_term_tax_rate": SHORT_TERM_TAX,
        "long_term_benchmark_tax_rate": LONG_TERM_TAX,
        "hashes": hashes,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    pct = lambda x: "n.a." if pd.isna(x) else f"{100 * float(x):.2f}%"
    num = lambda x: "n.a." if pd.isna(x) else f"{float(x):.3f}"
    integer = lambda x: str(int(x))
    top = strategy_summary.head(12).copy()
    quality_show = quality.sort_values(["mean_model_gross_return", "signals_complete"], ascending=[False, False]).head(15)
    strength_show = strength.sort_values(["horizon_days", "strength"])
    trade_show = best_trades[["centered_month", "information_month", "entry_date", "exit_date", "ONI", "dONI", "direction_label", "gross_return", "net_pre_tax_return", "regime"]]

    positive_regime = best_regimes[best_regimes["mean_net_trade_return"] > 0]
    regime_sentence = "No entry regime had a positive average net trade return."
    if len(positive_regime):
        leader = positive_regime.iloc[0]
        regime_sentence = f"The highest average occurred in {leader['regime'].lower()} ({pct(leader['mean_net_trade_return'])} per trade, {int(leader['trades'])} trades)."

    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Strong El Nino COP/XLE walk-forward simulation</title><style>
:root{{--bg:#07111d;--card:#101e2e;--text:#eef5fb;--muted:#a6b5c7;--line:#2b3d52;--blue:#38bdf8;--green:#34d399;--amber:#fbbf24;--red:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07101b,#0b1928);color:var(--text);font:15px/1.58 Inter,system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:44px 24px 72px}}h1{{font-size:42px;line-height:1.08;letter-spacing:-.03em;max-width:940px}}h2{{font-size:25px;margin-top:44px}}h3{{font-size:18px;margin-top:28px}}p,li{{max-width:940px}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:760;letter-spacing:.14em;text-transform:uppercase}}.lede{{font-size:19px;color:#cad7e5}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:24px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px}}.metric{{font-size:27px;font-weight:760}}.label,.small{{color:var(--muted);font-size:13px}}.callout{{border-left:4px solid var(--amber);padding:14px 18px;background:#1d1d20;border-radius:0 9px 9px 0;margin:18px 0}}.warning{{border-left-color:var(--red)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:14px 0 24px}}table{{border-collapse:collapse;width:100%;min-width:900px;background:var(--card)}}table.heat{{min-width:760px}}th{{text-align:left;background:#162a40;color:#c6d7e9;font-size:12px;text-transform:uppercase}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}svg{{display:block;width:100%;max-width:1080px;border:1px solid var(--line);border-radius:12px;background:#0b1522;margin:18px 0}}code{{background:#142438;padding:2px 5px;border-radius:4px}}a{{color:#7dd3fc}}footer{{border-top:1px solid var(--line);margin-top:44px;padding-top:18px;color:var(--muted)}}@media(max-width:820px){{.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:34px}}}}@media(max-width:520px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><div class='eyebrow'>Walk-forward simulation · test period 2020 to {last_date.date()}</div><h1>Strong El Niño signals and COP relative to XLE</h1><p class='lede'>The test uses only positive strong-El-Niño observations. Each centered ONI value is embargoed for two months, the direction is learned from completed earlier observations, and the frozen decision trades the next available interval. Signal quality and realized strategy P&amp;L are reported separately.</p>
<div class='cards'><div class='card'><div class='metric'>{html.escape(str(best['rule']).replace('S1 ','').replace('S2 ','').replace('S3 ','').replace('S4 ','').replace('S5 ',''))}</div><div class='label'>Highest after-tax rule at {int(best['horizon_days'])} trading days</div></div><div class='card'><div class='metric'>{pct(best['after_tax_cumulative_return'])}</div><div class='label'>Best simulated after-tax cumulative return</div></div><div class='card'><div class='metric'>{int(best['trades'])}</div><div class='label'>Non-overlapping trades in the best simulation</div></div><div class='card'><div class='metric'>{pct(benchmark_after_tax)}</div><div class='label'>XLE buy-and-hold after-tax return</div></div></div>
<div class='callout warning'><b>Interpretation.</b> This period contains one completed strong El Niño episode, 2023–24, plus a late 2026 signal with limited forward price coverage. Monthly signal observations within one episode are not independent. Results are descriptive and cannot establish repeatability across El Niño cycles.</div>
<h2>Simulation result</h2><p>The best combination by after-tax cumulative return was <b>{html.escape(str(best['rule']))}</b> with a <b>{int(best['horizon_days'])}-trading-day</b> holding period. It produced {pct(best['gross_cumulative_return'])} gross, {pct(best['net_pre_tax_cumulative_return'])} after commissions and before tax, and {pct(best['after_tax_cumulative_return'])} after the simplified tax model. Passive XLE returned {pct(benchmark_after_tax)} after tax over the full test period. XLE returned {pct(benchmark.iloc[1]['after_tax_return'])} after tax when held only during the best strategy's two windows. The continuous benchmark and intermittent strategy have different exposure.</p>{equity_chart}<h3>After-tax return across all 35 combinations</h3>{heatmap}
<h2>Top strategy and horizon combinations</h2>{table(top,[('rule','Signal rule',str),('horizon_days','Hold',lambda x:f'{int(x)}d'),('trades','Trades',integer),('gross_cumulative_return','Gross',pct),('net_pre_tax_cumulative_return','After commissions',pct),('after_tax_cumulative_return','After tax',pct),('mean_net_trade_return','Mean trade',pct),('win_rate','Win rate',pct),('max_closed_trade_drawdown','Max drawdown',pct),('tax_paid','Tax paid',lambda x:f'${float(x):,.0f}')])}
<h2>Signal quality</h2><p>Signal quality uses every completed walk-forward forecast, including observations that overlap another trade. It measures whether the frozen direction was correct before commissions and tax. Strategy P&amp;L below uses non-overlapping trades only. The displayed p-values are unadjusted descriptive tests; overlap and the single completed climate episode make them unsuitable as proof of statistical significance.</p>{table(quality_show,[('rule','Signal rule',str),('horizon_days','Horizon',lambda x:f'{int(x)}d'),('signals_complete','Complete signals',integer),('unique_episodes','Approx. episodes',integer),('mean_spread_return','Mean COP − XLE',pct),('mean_model_gross_return','Mean model gross',pct),('directional_hit_rate','Directional hit',pct),('signal_return_r','ONI-return r',num),('signal_return_p','Correlation p',num),('mean_p','Mean-return p',num)])}
<h3>Results by signal strength</h3>{table(strength_show,[('strength','Frozen ONI strength',str),('horizon_days','Horizon',lambda x:f'{int(x)}d'),('n','N',integer),('mean_spread_return','Mean COP − XLE',pct),('mean_model_gross_return','Mean model gross',pct),('directional_hit_rate','Hit rate',pct),('p','Mean-return p',num)])}
<h2>Best-strategy trade audit</h2>{table(trade_show,[('centered_month','Centered ONI month',str),('information_month','Usable month',str),('entry_date','Entry',lambda x:str(pd.Timestamp(x).date())),('exit_date','Exit',lambda x:str(pd.Timestamp(x).date())),('ONI','ONI',lambda x:f'{float(x):.1f}'),('dONI','ΔONI',lambda x:f'{float(x):+.1f}'),('direction_label','Frozen position',str),('gross_return','Gross',pct),('net_pre_tax_return','After commission',pct),('regime','Entry regime',str)])}
<h2>Market regimes</h2><p>At each entry, the report classifies the energy market using only prior XLE data. An uptrend means XLE's trailing 126-session total return was positive. High volatility means trailing 20-session annualized volatility was above its expanding historical median. {html.escape(regime_sentence)} All regime findings have very small samples.</p>{table(best_regimes,[('regime','Entry regime',str),('trades','Trades',integer),('mean_net_trade_return','Mean trade',pct),('win_rate','Win rate',pct),('cumulative_net_pre_tax_return','Cumulative',pct)])}
<h2>Energy benchmark and tax sensitivity</h2>{table(benchmark,[('benchmark','Benchmark',str),('entry_date','Entry',lambda x:str(pd.Timestamp(x).date())),('exit_date','Exit',lambda x:str(pd.Timestamp(x).date())),('gross_return','Gross',pct),('net_pre_tax_return','After commissions',pct),('tax_rate','Tax rate',pct),('tax_paid','Tax paid',lambda x:f'${float(x):,.0f}'),('after_tax_return','After tax',pct)])}{table(tax_sensitivity,[('case','Case',str),('tax_rate','Tax rate',pct),('after_tax_return','Return after tax',pct),('tax_paid','Tax paid',lambda x:f'${float(x):,.0f}')])}
<h2>Five signal specifications</h2><ul><li><b>S1 Strong level:</b> frozen ONI ≥ 1.0.</li><li><b>S2 Very strong:</b> ONI ≥ 1.5.</li><li><b>S3 Strong and rising:</b> ONI ≥ 1.0 and ΔONI &gt; 0.</li><li><b>S4 Rapid strengthening:</b> ONI ≥ 1.0 and ΔONI ≥ 0.3.</li><li><b>S5 Strong but cooling:</b> ONI remains ≥ 1.0 but ΔONI &lt; 0. This is a post-strengthening signal, not an ex-post declaration of the exact peak.</li></ul>
<h2>Walk-forward mechanics</h2><ol><li>A centered ONI observation for month <i>m</i> becomes eligible only after month <i>m+2</i>. The trade begins at the next common COP/XLE session's open.</li><li>For each rule and holding period, training uses only earlier qualifying observations whose exits occurred before the new entry date.</li><li>With at least {MIN_TRAIN} completed training observations, the sign of the historical mean COP-minus-XLE return sets the position. Negative mean produces long XLE/short COP; positive mean produces long COP/short XLE.</li><li>The direction, rule and horizon are frozen for that trade. A strategy does not open a new trade while its preceding trade remains active.</li></ol>
<h2>Return, commission and tax formulas</h2><p>The capital-normalized pair return is <code>0.5 × long-leg total return − 0.5 × short-leg total return</code>. Each leg receives 50% notional, producing 100% gross exposure and zero initial net market value. Massive split-adjusted prices are supplemented with split-adjusted cash dividends; the short leg owes dividends during the holding period.</p><p>Commission is 10 basis points per order notional. Two 50% legs traded at entry and exit produce a 20-basis-point round-trip cost on capital: <code>2 × 0.10% = 0.20%</code>. Net pre-tax return is <code>gross return − 0.20%</code>.</p><p>The base personal-tax illustration assumes a US taxable individual. It nets realized trade profit and loss within each calendar year and charges 37% on a positive annual net short-term result. It assumes no state tax, 3.8% net-investment-income tax, loss carryforward, borrowing cost, short-borrow fee, margin interest or slippage. For simplicity, the XLE buy-and-hold illustration applies a 20% rate to its positive total return, including dividends, at the end of the test. Actual dividend and capital-gain timing will differ. Tax results are scenario estimates, not personal tax advice.</p>
<h2>Limitations</h2><ul><li>The supplied ONI series is revised history. The two-month embargo prevents premature use of recent values but does not reconstruct exact NOAA vintages.</li><li>The test has effectively one completed strong event. Counts of monthly signals overstate the number of independent climate episodes.</li><li>Choosing the best of 35 rule-and-horizon combinations introduces selection bias. No untouched second strong event exists inside 2020–2026 for final validation.</li><li>Daily closes and dividends come from Massive. The execution model uses next-session open and horizon-session close, but it does not include bid-ask spreads, locate constraints or market impact.</li><li>COP/XLE is a relative trade. It is not equivalent to a directional investment in COP or XLE.</li></ul>
<h2>Sources</h2><ul><li><a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>Massive stock aggregate documentation</a></li><li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/'>NOAA CPC ONI table and revision note</a></li><li><a href='https://www.irs.gov/taxtopics/tc409'>IRS Topic 409: capital gains and losses</a></li><li><a href='https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill'>IRS 2026 individual tax rates</a></li></ul>
<footer>Last common Massive price date: {last_date.date()}. Initial simulated capital: ${INITIAL_CAPITAL:,.0f}. All detailed observations, trades, summaries and data hashes are saved beside this report.</footer></main></body></html>"""
    (ROOT / "COP_XLE_strong_El_Nino_walkforward_report.html").write_text(report)
    print(strategy_summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
