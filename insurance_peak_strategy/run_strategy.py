#!/usr/bin/env python3
"""Peak-El-Nino technical/short-pressure strategy simulation."""

from __future__ import annotations

import base64
import gzip
import html
import io
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
SOURCE_RAW = ROOT.parent / "insurance_oni_event_study" / "raw"
ONI_SOURCE = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(ROOT.parent / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
CALC = ROOT / "calculated"
REPORT = ROOT / "peak_el_nino_insurance_strategy.html"
LATEST_REPORT = ROOT / "latest_event_report.html"
TICKERS = ("ROOT", "PLMR", "KNSL", "MCY", "LMND", "CINF", "PGR", "AIZ", "AIG", "BAP", "ALL", "HIG")
INITIAL_CAPITAL = 100_000.0
ATR_DAYS = 5
ATR_MULTIPLE = 2
COST_PER_SIDE = 0.00075
CAPITAL_GAINS_TAX_RATE = 0.20
SHORT_BORROW_ANNUAL = 0.03
ONI_THRESHOLD = 0
ONI_PUBLICATION_SHIFT = 3


def load_price(ticker: str) -> pd.DataFrame:
    matches = sorted(SOURCE_RAW.glob(f"massive_{ticker}_daily_*.json.gz"))
    if not matches:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with gzip.open(matches[-1], "rt", encoding="utf-8") as handle:
        rows = json.load(handle).get("results", [])
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame({
        "date": pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert(None).normalize(),
        "open": [float(r["o"]) for r in rows],
        "high": [float(r["h"]) for r in rows],
        "low": [float(r["l"]) for r in rows],
        "close": [float(r["c"]) for r in rows],
        "volume": [float(r.get("v", 0)) for r in rows],
    }).drop_duplicates("date", keep="last").set_index("date").sort_index()
    return frame


def load_json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def oni_episodes() -> list[dict]:
    frame = pd.read_csv(ONI_SOURCE)
    frame["month"] = pd.to_datetime(frame["Date"]).dt.to_period("M")
    oni = frame.set_index("month")["ONI"].astype(float)
    severe = oni[oni >= ONI_THRESHOLD]
    groups: list[list[pd.Period]] = []
    for month in severe.index:
        if not groups or month.ordinal != groups[-1][-1].ordinal + 1:
            groups.append([month])
        else:
            groups[-1].append(month)
    out = []
    for months in groups:
        if months[-1] < pd.Period("2009-01", "M"):
            continue
        trade_start = months[0] + ONI_PUBLICATION_SHIFT
        trade_end = months[-1] + ONI_PUBLICATION_SHIFT
        out.append({
            "episode": f"{months[0]} to {months[-1]}",
            "peak_oni": float(oni.loc[months].max()),
            "signal_start": str(months[0]),
            "signal_end": str(months[-1]),
            "trade_start": trade_start.to_timestamp(how="start"),
            "trade_end": trade_end.to_timestamp(how="end").normalize(),
            "partial": months[-1] == oni.index.max(),
        })
    return out


def load_alt(ticker: str, price_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    alt = pd.DataFrame(index=price_index)
    coverage: dict[str, object] = {"ticker": ticker}

    si_rows = load_json_rows(ROOT / "raw" / "short_interest" / f"{ticker}.json.gz")
    if si_rows:
        si = pd.DataFrame(si_rows)
        si["settlement_date"] = pd.to_datetime(si["settlement_date"])
        si["available_date"] = si["settlement_date"] + pd.Timedelta(days=10)
        si = si.sort_values("available_date").set_index("available_date")
        si["si_change"] = si["short_interest"].pct_change()
        si_daily = si[["short_interest", "days_to_cover", "si_change"]].reindex(price_index, method="ffill")
        alt = alt.join(si_daily)
        coverage.update(si_start=str(si["settlement_date"].min().date()), si_end=str(si["settlement_date"].max().date()), si_rows=len(si))
    else:
        alt[["short_interest", "days_to_cover", "si_change"]] = np.nan
        coverage.update(si_start="", si_end="", si_rows=0)

    sv_rows = load_json_rows(ROOT / "raw" / "short_volume" / f"{ticker}.json.gz")
    if sv_rows:
        sv = pd.DataFrame(sv_rows)
        sv["date"] = pd.to_datetime(sv["date"])
        sv = sv.drop_duplicates("date", keep="last").set_index("date").sort_index()
        ratio = sv["short_volume_ratio"].astype(float)
        alt["short_volume_ratio"] = ratio.reindex(price_index)
        coverage.update(sv_start=str(sv.index.min().date()), sv_end=str(sv.index.max().date()), sv_rows=len(sv))
    else:
        alt["short_volume_ratio"] = np.nan
        coverage.update(sv_start="", sv_end="", sv_rows=0)

    quote_rows = []
    for path in sorted((ROOT / "raw" / "quotes_weekly_nbbo" / ticker).glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("results", []):
            total_size = float(row.get("bid_size", 0)) + float(row.get("ask_size", 0))
            mid = (float(row.get("bid_price", 0)) + float(row.get("ask_price", 0))) / 2
            quote_rows.append({
                "quote_date": pd.Timestamp(payload["session_date"]),
                "book_imbalance": (float(row.get("bid_size", 0)) - float(row.get("ask_size", 0))) / total_size if total_size else np.nan,
                "spread_bps": (float(row.get("ask_price", 0)) - float(row.get("bid_price", 0))) / mid * 10000 if mid else np.nan,
            })
    if quote_rows:
        q = pd.DataFrame(quote_rows).drop_duplicates("quote_date", keep="last").sort_values("quote_date")
        # A Friday close quote becomes usable only on the next trading session.
        q["available_date"] = q["quote_date"] + pd.Timedelta(days=1)
        q = q.set_index("available_date")
        alt = alt.join(q[["book_imbalance", "spread_bps"]].reindex(price_index, method="ffill"))
        coverage.update(book_start=str(q["quote_date"].min().date()), book_end=str(q["quote_date"].max().date()), book_rows=len(q))
    else:
        alt[["book_imbalance", "spread_bps"]] = np.nan
        coverage.update(book_start="", book_end="", book_rows=0)
    return alt, coverage


def reference_snapshot(ticker: str, trade_start: pd.Timestamp) -> dict:
    stamp = trade_start.strftime("%Y-%m-01")
    rows = load_json_rows(ROOT / "raw" / "reference" / f"{ticker}_{stamp}.json.gz")
    if not rows:
        return {}
    row = rows[0]
    return {
        "shares_outstanding_proxy": row.get("share_class_shares_outstanding"),
        "weighted_shares": row.get("weighted_shares_outstanding"),
        "market_cap": row.get("market_cap"),
    }


def features(ticker: str, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = prices.copy()
    prev = frame["close"].shift(1)
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev).abs(),
        (frame["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    frame["atr10"] = tr.ewm(alpha=1 / ATR_DAYS, adjust=False, min_periods=ATR_DAYS).mean()
    frame["atr_pct"] = frame["atr10"] / frame["close"]
    frame["atr60_median"] = frame["atr_pct"].rolling(60, min_periods=40).median()
    frame["sma20"] = frame["close"].rolling(20).mean()
    frame["sma60"] = frame["close"].rolling(60).mean()
    frame["ret5"] = np.log(frame["close"]).diff(5)
    frame["trend"] = np.select(
        [(frame["close"] > frame["sma20"]) & (frame["sma20"] > frame["sma60"]) & (frame["ret5"] > 0),
         (frame["close"] < frame["sma20"]) & (frame["sma20"] < frame["sma60"]) & (frame["ret5"] < 0)],
        [1.0, -1.0], default=0.0,
    )
    alt, coverage = load_alt(ticker, frame.index)
    frame = frame.join(alt)
    frame["book_signal"] = np.select([frame["book_imbalance"] > .10, frame["book_imbalance"] < -.10], [1.0, -1.0], default=0.0)
    frame.loc[frame["book_imbalance"].isna(), "book_signal"] = np.nan
    frame["si_signal"] = np.select([frame["si_change"] < -.05, frame["si_change"] > .05], [1.0, -1.0], default=0.0)
    frame.loc[frame["si_change"].isna(), "si_signal"] = np.nan
    frame["sv5"] = frame["short_volume_ratio"].rolling(5, min_periods=3).mean()
    frame["sv20_median"] = frame["short_volume_ratio"].rolling(20, min_periods=10).median()
    frame["sv_signal"] = np.select([frame["sv5"] < frame["sv20_median"] - 3, frame["sv5"] > frame["sv20_median"] + 3], [1.0, -1.0], default=0.0)
    frame.loc[frame["sv20_median"].isna(), "sv_signal"] = np.nan
    components = [("trend", .55), ("book_signal", .20), ("si_signal", .15), ("sv_signal", .10)]
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for name, weight in components:
        available = frame[name].notna()
        numerator = numerator.add(frame[name].fillna(0) * weight)
        denominator = denominator.add(available.astype(float) * weight)
    frame["score"] = numerator / denominator.replace(0, np.nan)
    vol_high = frame["atr_pct"] > frame["atr60_median"]
    frame["core_signal"] = np.where(vol_high, frame["trend"], 0.0)
    confirmed = (frame["trend"] != 0) & (frame["score"] * frame["trend"] >= .55) & vol_high
    frame["enhanced_signal"] = np.where(confirmed, frame["trend"], 0.0)
    return frame, coverage


def simulate(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, signal_column: str) -> tuple[pd.DataFrame, list[dict]]:
    data = frame.loc[:end].copy()
    event_sessions = data[(data.index >= start) & (data.index <= end)].index
    if len(event_sessions) < 2:
        return pd.DataFrame(), []
    first, last = event_sessions[0], event_sessions[-1]
    rows, trades = [], []
    pos = 0
    stop = math.nan
    entry_date = None
    entry_price = math.nan
    cooldown = 0
    prior_signal = 0
    dates = data.index
    start_loc = dates.get_loc(first)
    for loc in range(start_loc, dates.get_loc(last) + 1):
        date = dates[loc]
        row = data.iloc[loc]
        prev = data.iloc[loc - 1]
        raw_signal = int(prev[signal_column]) if pd.notna(prev[signal_column]) else 0
        if raw_signal != prior_signal:
            cooldown = 0
        desired = 0 if cooldown == raw_signal and raw_signal != 0 else raw_signal
        old_pos = pos
        overnight = old_pos * (row["open"] / prev["close"] - 1) if old_pos else 0.0
        cost = 0.0
        turnover = 0
        if desired != old_pos:
            turnover += abs(desired - old_pos)
            cost += abs(desired - old_pos) * COST_PER_SIDE
            if old_pos:
                trades[-1]["exit_date"] = date
                trades[-1]["exit_reason"] = "signal"
            pos = desired
            if pos:
                entry_date, entry_price = date, float(row["open"])
                atr_ref = float(prev["atr10"])
                stop = entry_price - pos * ATR_MULTIPLE * atr_ref
                trades.append({"entry_date": date, "side": "long" if pos > 0 else "short", "entry_price": entry_price, "exit_date": pd.NaT, "exit_reason": ""})
            else:
                stop = math.nan
        intraday = 0.0
        stopped = False
        exit_price = float(row["close"])
        if pos:
            if pos > 0 and row["low"] <= stop:
                exit_price = float(row["open"] if row["open"] <= stop else stop)
                stopped = True
            elif pos < 0 and row["high"] >= stop:
                exit_price = float(row["open"] if row["open"] >= stop else stop)
                stopped = True
            intraday = pos * (exit_price / row["open"] - 1)
        factor = (1 + overnight) * (1 + intraday)
        daily_return = factor - 1 - cost
        if pos < 0:
            daily_return -= SHORT_BORROW_ANNUAL / 252
        if stopped:
            daily_return -= COST_PER_SIDE
            turnover += 1
            trades[-1]["exit_date"] = date
            trades[-1]["exit_reason"] = "ATR stop"
            trades[-1]["exit_price"] = exit_price
            cooldown = pos
            pos = 0
            stop = math.nan
        elif pos:
            atr_now = float(row["atr10"])
            stop = max(stop, float(row["close"]) - ATR_MULTIPLE * atr_now) if pos > 0 else min(stop, float(row["close"]) + ATR_MULTIPLE * atr_now)

        if date == last and pos:
            daily_return -= COST_PER_SIDE
            turnover += 1
            trades[-1]["exit_date"] = date
            trades[-1]["exit_reason"] = "event end"
            trades[-1]["exit_price"] = float(row["close"])
            pos = 0
            stop = math.nan
        rows.append({
            "date": date, "return": daily_return, "position": old_pos if desired == old_pos else desired,
            "signal_used": raw_signal, "score_used": float(prev["score"]) if pd.notna(prev["score"]) else np.nan,
            "turnover": turnover, "stop_hit": stopped,
        })
        prior_signal = raw_signal
    return pd.DataFrame(rows).set_index("date"), trades


def metrics(returns: pd.Series, exposure: pd.Series | None = None, trades: int = 0, stops: int = 0) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"return": np.nan, "ending_capital": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "active_day_hit": np.nan, "exposure": np.nan, "trades": trades, "stops": stops}
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    active = r[r != 0]
    return {
        "return": float(wealth.iloc[-1] - 1), "ending_capital": float(INITIAL_CAPITAL * wealth.iloc[-1]),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(252)),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if r.std(ddof=1) else np.nan,
        "max_drawdown": float(dd.min()), "active_day_hit": float((active > 0).mean()) if len(active) else np.nan,
        "exposure": float(exposure.abs().mean()) if exposure is not None and len(exposure) else np.nan,
        "trades": trades, "stops": stops,
    }


def passive_returns(prices: dict[str, pd.DataFrame], names: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return fixed equal-dollar buy-and-hold performance after costs and tax."""
    wealth_panel = {}
    for ticker in names:
        close = prices[ticker]["close"]
        event_close = close[(close.index >= start) & (close.index <= end)]
        if len(event_close) < 2:
            continue
        before = close[close.index < event_close.index[0]]
        entry = pd.concat([before.tail(1), event_close])
        wealth_panel[ticker] = entry / entry.iloc[0]
    if not wealth_panel:
        return pd.Series(dtype=float)

    wealth = pd.DataFrame(wealth_panel).iloc[1:].mean(axis=1)
    gross_gain = float(wealth.iloc[-1] - 1)
    total_cost = 2 * COST_PER_SIDE
    tax = max(gross_gain - total_cost, 0.0) * CAPITAL_GAINS_TAX_RATE
    net_wealth = wealth * (1 - total_cost)
    net_wealth.iloc[-1] -= tax
    return net_wealth.pct_change().fillna(net_wealth.iloc[0] - 1)


def fmt_pct(x, digits=1):
    return "n.a." if pd.isna(x) else f"{100*x:.{digits}f}%"


def fmt_num(x, digits=2):
    return "n.a." if pd.isna(x) else f"{x:.{digits}f}"


def table_html(frame: pd.DataFrame, columns: list[tuple[str, str, str]], ident: str) -> str:
    heads = "".join(f"<th onclick=\"sortTable('{ident}',{i})\">{html.escape(label)}</th>" for i, (_, label, _) in enumerate(columns))
    out = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            shown = fmt_pct(value) if kind == "pct" else fmt_num(value) if kind == "num" else f"${value:,.0f}" if kind == "usd" and pd.notna(value) else str(int(value)) if kind == "int" and pd.notna(value) else html.escape(str(value)) if pd.notna(value) else "n.a."
            cls = "neg" if kind == "pct" and pd.notna(value) and value < 0 else "pos" if kind == "pct" and pd.notna(value) and value > 0 else ""
            cells.append(f"<td class='{cls}'>{shown}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table id='{ident}'><thead><tr>{heads}</tr></thead><tbody>{''.join(out)}</tbody></table></div>"


def chart_data(fig) -> str:
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def main() -> None:
    CALC.mkdir(parents=True, exist_ok=True)
    episodes = oni_episodes()
    completed = [e for e in episodes if e["trade_end"] <= pd.Timestamp("2026-09-11")]
    prices = {ticker: load_price(ticker) for ticker in (*TICKERS, "KIE", "SPY")}
    feature_frames, coverage_rows = {}, []
    for ticker in TICKERS:
        feature_frames[ticker], coverage = features(ticker, prices[ticker])
        coverage_rows.append(coverage)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(CALC / "alternative_data_coverage.csv", index=False)

    event_summary_rows, ticker_rows, daily_portfolios, trade_rows, tendency_rows, quality_rows = [], [], [], [], [], []
    for ep in completed:
        event_id = ep["episode"]
        start, end = ep["trade_start"], ep["trade_end"]
        strategy_daily = {"Core MA + ATR": {}, "Enhanced short/order-book": {}}
        position_daily = {"Core MA + ATR": {}, "Enhanced short/order-book": {}}
        available_tickers = []
        for ticker in TICKERS:
            frame = feature_frames[ticker]
            event_slice = frame[(frame.index >= start) & (frame.index <= end)]
            if len(event_slice) < 5 or frame.loc[:start].shape[0] < 60:
                continue
            available_tickers.append(ticker)
            for label, signal_col in (("Core MA + ATR", "core_signal"), ("Enhanced short/order-book", "enhanced_signal")):
                sim, trades = simulate(frame, start, end, signal_col)
                if sim.empty:
                    continue
                strategy_wealth = (1 + sim["return"]).cumprod()
                strategy_tax = max(float(strategy_wealth.iloc[-1] - 1), 0.0) * CAPITAL_GAINS_TAX_RATE
                sim.loc[sim.index[-1], "return"] -= strategy_tax / strategy_wealth.iloc[-2] if len(strategy_wealth) > 1 else strategy_tax
                strategy_daily[label][ticker] = sim["return"]
                position_daily[label][ticker] = sim["position"]
                m = metrics(sim["return"], sim["position"], len(trades), int(sim["stop_hit"].sum()))
                ticker_rows.append({"episode": event_id, "ticker": ticker, "strategy": label, **m})
                for trade in trades:
                    trade_rows.append({"episode": event_id, "ticker": ticker, "strategy": label, **trade})

            # Signal quality uses close-t information versus t+1 close-to-close return.
            z = event_slice.copy()
            z["next_return"] = np.log(frame["close"]).diff().shift(-1).reindex(z.index)
            q = z[["score", "enhanced_signal", "next_return"]].dropna()
            active = q[q["enhanced_signal"] != 0]
            ic, ic_p = (stats.pearsonr(q["score"], q["next_return"]) if len(q) >= 8 and q["score"].std() else (np.nan, np.nan))
            quality_rows.append({
                "episode": event_id, "ticker": ticker, "observations": len(q), "active_signals": len(active),
                "score_next_day_ic": float(ic), "ic_p": float(ic_p),
                "direction_hit_rate": float((np.sign(active["enhanced_signal"]) == np.sign(active["next_return"])).mean()) if len(active) else np.nan,
                "mean_signed_next_day": float((active["enhanced_signal"] * active["next_return"]).mean()) if len(active) else np.nan,
            })

            ref = reference_snapshot(ticker, start)
            latest_si = event_slice["short_interest"].dropna()
            shares = ref.get("weighted_shares") or ref.get("shares_outstanding_proxy")
            tendency_rows.append({
                "episode": event_id, "ticker": ticker, "sessions": len(event_slice),
                "annualized_realized_vol": float(np.log(event_slice["close"]).diff().std(ddof=1) * math.sqrt(252)),
                "mean_atr_pct": float(event_slice["atr_pct"].mean()),
                "trend_up_share": float((event_slice["trend"] > 0).mean()), "trend_down_share": float((event_slice["trend"] < 0).mean()),
                "mean_book_imbalance": float(event_slice["book_imbalance"].mean()), "median_spread_bps": float(event_slice["spread_bps"].median()),
                "mean_short_volume_ratio": float(event_slice["short_volume_ratio"].mean()),
                "last_short_interest": float(latest_si.iloc[-1]) if len(latest_si) else np.nan,
                "shares_outstanding_proxy": shares,
                "short_interest_pct_shares": float(latest_si.iloc[-1] / shares) if len(latest_si) and shares else np.nan,
            })

        for label in strategy_daily:
            returns_panel = pd.DataFrame(strategy_daily[label])
            pos_panel = pd.DataFrame(position_daily[label])
            portfolio_return = returns_panel.mean(axis=1, skipna=True)
            portfolio_pos = pos_panel.abs().mean(axis=1, skipna=True)
            m = metrics(portfolio_return, portfolio_pos,
                        int(sum((r["strategy"] == label and r["episode"] == event_id) * r["trades"] for r in ticker_rows)),
                        int(sum((r["strategy"] == label and r["episode"] == event_id) * r["stops"] for r in ticker_rows)))
            event_summary_rows.append({"episode": event_id, "trade_start": start.date(), "trade_end": end.date(), "series": label, "tickers": len(returns_panel.columns), **m})
            for date, value in portfolio_return.items():
                daily_portfolios.append({"episode": event_id, "date": date, "series": label, "return": value})

        # Fixed equal-dollar buy-and-hold and passive benchmarks.
        for label, names in (
            ("Passive insurance basket", available_tickers),
            ("US P&C insurance benchmark (KIE proxy)", ["KIE"]),
            ("SPY", ["SPY"]),
        ):
            r = passive_returns(prices, names, start, end)
            m = metrics(r)
            event_summary_rows.append({"episode": event_id, "trade_start": start.date(), "trade_end": end.date(), "series": label, "tickers": len(names), **m})
            for date, value in r.items():
                daily_portfolios.append({"episode": event_id, "date": date, "series": label, "return": value})

    event_summary = pd.DataFrame(event_summary_rows)
    ticker_results = pd.DataFrame(ticker_rows)
    daily = pd.DataFrame(daily_portfolios)
    trades = pd.DataFrame(trade_rows)
    tendencies = pd.DataFrame(tendency_rows)
    signal_quality = pd.DataFrame(quality_rows)
    event_summary.to_csv(CALC / "event_strategy_summary.csv", index=False)
    ticker_results.to_csv(CALC / "ticker_strategy_results.csv", index=False)
    daily.to_csv(CALC / "daily_portfolio_returns.csv", index=False)
    trades.to_csv(CALC / "trade_log.csv", index=False)
    tendencies.to_csv(CALC / "ticker_event_tendencies.csv", index=False)
    signal_quality.to_csv(CALC / "signal_quality.csv", index=False)

    # Overall series link event paths multiplicatively; gaps carry cash.
    overall_rows = []
    for series, group in daily.groupby("series"):
        r = group.sort_values("date").set_index("date")["return"]
        overall_rows.append({"series": series, **metrics(r)})
    overall = pd.DataFrame(overall_rows).sort_values("return", ascending=False)
    overall.to_csv(CALC / "overall_strategy_summary.csv", index=False)

    # Figures.
    event_charts = []
    chart_series = ("Enhanced short/order-book", "Core MA + ATR", "Passive insurance basket", "US P&C insurance benchmark (KIE proxy)", "SPY")
    for event_id, group in daily.groupby("episode", sort=True):
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for series in chart_series:
            series_rows = group[group["series"] == series].sort_values("date")
            if len(series_rows):
                wealth = INITIAL_CAPITAL * (1 + series_rows["return"]).cumprod()
                ax.plot(series_rows["date"], wealth, label=series, lw=2 if "Enhanced" in series else 1.3)
        ax.axhline(INITIAL_CAPITAL, color="#64748b", lw=.8, ls="--")
        ax.set(title=f"Returns during tradable event window: {event_id}", ylabel="Portfolio value (USD)")
        ax.grid(alpha=.2); ax.legend(frameon=False, ncol=2)
        slug = event_id.replace(" ", "_").replace("/", "-")
        chart_path = CALC / f"returns_{slug}.png"
        fig.savefig(chart_path, dpi=160, bbox_inches="tight", facecolor="white")
        event_charts.append((event_id, chart_data(fig)))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for series in chart_series:
        g = daily[daily["series"] == series].sort_values("date")
        if len(g):
            wealth = INITIAL_CAPITAL * (1 + g["return"]).cumprod()
            ax.plot(g["date"], wealth, label=series, lw=2 if "Enhanced" in series else 1.3)
    ax.set(title="Capital across tradable peak-signal windows", ylabel="Portfolio value (USD)")
    ax.grid(alpha=.2); ax.legend(frameon=False, ncol=2)
    chart_wealth = chart_data(fig)

    enhanced_tickers = ticker_results[ticker_results["strategy"] == "Enhanced short/order-book"].groupby("ticker")["return"].apply(lambda x: float(np.prod(1 + x) - 1)).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(enhanced_tickers.index, enhanced_tickers.values, color=np.where(enhanced_tickers.values < 0, "#b91c1c", "#15803d"))
    ax.axvline(0, color="#475569", lw=.8); ax.set(title="Enhanced strategy return by ticker across available episodes", xlabel="Cumulative return")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}"); ax.grid(axis="x", alpha=.2)
    chart_tickers = chart_data(fig)

    qplot = signal_quality.groupby("ticker").agg(hit=("direction_hit_rate", "mean"), ic=("score_next_day_ic", "mean"), n=("active_signals", "sum")).sort_values("hit")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(qplot.index, qplot["hit"], color="#2563eb"); ax.axvline(.5, color="#b91c1c", ls="--", lw=1)
    ax.set(xlim=(0, 1), title="Directional signal quality on active days", xlabel="Mean next-day sign hit rate")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}"); ax.grid(axis="x", alpha=.2)
    chart_quality = chart_data(fig)

    overall_table = table_html(overall, [("series","Series","text"),("return","Cumulative return","pct"),("ending_capital","Ending capital","usd"),("ann_vol","Annualized vol","pct"),("sharpe","Sharpe","num"),("max_drawdown","Max drawdown","pct")], "overall")
    event_table = table_html(event_summary.sort_values(["episode","series"]), [("episode","Actual peak episode","text"),("trade_start","Trade start","text"),("trade_end","Trade end","text"),("series","Series","text"),("tickers","Tickers","int"),("return","Return","pct"),("ann_vol","Ann. vol","pct"),("sharpe","Sharpe","num"),("max_drawdown","Max drawdown","pct"),("exposure","Exposure","pct"),("trades","Trades","int"),("stops","ATR stops","int")], "events")
    ticker_table = table_html(ticker_results[ticker_results["strategy"] == "Enhanced short/order-book"].sort_values(["episode","return"]), [("episode","Episode","text"),("ticker","Ticker","text"),("return","Return","pct"),("ending_capital","$100k equivalent","usd"),("ann_vol","Ann. vol","pct"),("sharpe","Sharpe","num"),("max_drawdown","Max drawdown","pct"),("active_day_hit","Active-day hit","pct"),("exposure","Exposure","pct"),("trades","Trades","int"),("stops","Stops","int")], "tickers")
    tendency_table = table_html(tendencies.sort_values(["episode","ticker"]), [("episode","Episode","text"),("ticker","Ticker","text"),("annualized_realized_vol","Realized vol","pct"),("mean_atr_pct","Mean ATR/price","pct"),("trend_up_share","Up-trend share","pct"),("trend_down_share","Down-trend share","pct"),("mean_book_imbalance","Mean book imbalance","num"),("median_spread_bps","Median spread (bp)","num"),("mean_short_volume_ratio","Short-volume ratio (%)","num"),("short_interest_pct_shares","SI/share proxy","pct")], "tendencies")
    quality_table = table_html(signal_quality.groupby("ticker",as_index=False).agg(observations=("observations","sum"),active_signals=("active_signals","sum"),score_next_day_ic=("score_next_day_ic","mean"),ic_p=("ic_p","mean"),direction_hit_rate=("direction_hit_rate","mean"),mean_signed_next_day=("mean_signed_next_day","mean")).sort_values("direction_hit_rate",ascending=False), [("ticker","Ticker","text"),("observations","Observations","int"),("active_signals","Active signals","int"),("score_next_day_ic","Score IC","num"),("ic_p","Mean IC p","num"),("direction_hit_rate","Direction hit","pct"),("mean_signed_next_day","Mean signed next-day","pct")], "quality")
    coverage_table = table_html(coverage, [("ticker","Ticker","text"),("si_start","Short interest start","text"),("si_end","SI end","text"),("si_rows","SI rows","int"),("sv_start","Short volume start","text"),("sv_end","SV end","text"),("sv_rows","SV rows","int"),("book_start","NBBO start","text"),("book_end","NBBO end","text"),("book_rows","NBBO weeks","int")], "coverage")

    enhanced = overall.set_index("series").loc["Enhanced short/order-book"]
    core = overall.set_index("series").loc["Core MA + ATR"]
    basket = overall.set_index("series").loc["Passive insurance basket"]
    kie = overall.set_index("series").loc["US P&C insurance benchmark (KIE proxy)"]
    best_ticker, best_ret = enhanced_tickers.index[-1], enhanced_tickers.iloc[-1]
    worst_ticker, worst_ret = enhanced_tickers.index[0], enhanced_tickers.iloc[0]
    future_episode = [e for e in episodes if e not in completed]
    future_note = f"The {future_episode[0]['episode']} peak first becomes tradable under the timing rule in {future_episode[0]['trade_start'].strftime('%B %Y')}; it is therefore not backtested." if future_episode else ""
    event_chart_html = "".join(
        f"<h3>{html.escape(event_id)}</h3><img class='chart' src='{chart}' alt='Returns during {html.escape(event_id)} event window'>"
        for event_id, chart in event_charts
    )

    css = """:root{--ink:#172033;--muted:#5f6b7a;--line:#dbe2ea;--navy:#0f2747;--blue:#2563eb;--red:#b91c1c;--green:#15803d;--wash:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.52 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}main{max-width:1220px;margin:30px auto;background:#fff;padding:48px 54px 70px;box-shadow:0 8px 30px #20304018}h1{margin:0;color:var(--navy);font-size:34px;line-height:1.12}h2{font-size:23px;color:var(--navy);margin:42px 0 12px;padding-top:27px;border-top:1px solid var(--line)}h3{font-size:17px;color:var(--navy)}.deck{font-size:18px;color:var(--muted);max-width:900px}.meta,.small{color:var(--muted);font-size:13px}.callout{border-left:5px solid var(--blue);background:#eff6ff;padding:18px 20px;margin:22px 0}.warning{border-left-color:#d97706;background:#fff7ed}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}.card{border:1px solid var(--line);padding:16px;border-radius:8px}.card b{display:block;font-size:24px;color:var(--navy)}img.chart{display:block;max-width:100%;margin:18px auto}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px;margin:14px 0 22px}table{border-collapse:collapse;width:100%;font-size:12.5px}th{background:var(--navy);color:#fff;padding:9px 8px;white-space:nowrap;cursor:pointer;position:sticky;top:0}td{padding:8px;border-bottom:1px solid #e8edf3;white-space:nowrap;text-align:right}td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}tr:nth-child(even){background:#f8fafc}.neg{color:var(--red)}.pos{color:var(--green)}code{background:#eef2f7;padding:2px 5px;border-radius:4px}li{margin:7px 0}a{color:var(--blue)}@media(max-width:760px){main{margin:0;padding:28px 20px}.grid{grid-template-columns:1fr}}"""
    js = """function sortTable(id,n){const t=document.getElementById(id),b=t.tBodies[0],r=[...b.rows];const asc=t.dataset.sortcol==n&&t.dataset.dir!='asc';r.sort((a,c)=>{let x=a.cells[n].innerText.replace(/[$,%<>]/g,''),y=c.cells[n].innerText.replace(/[$,%<>]/g,'');let nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;return asc?x.localeCompare(y):y.localeCompare(x)});r.forEach(x=>b.appendChild(x));t.dataset.sortcol=n;t.dataset.dir=asc?'asc':'desc'}"""
    report = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Peak El Niño insurance strategy</title><style>{css}</style></head><body><main>
    <h1>Peak El Niño insurance strategy</h1><p class='deck'>Point-in-time technical and short-pressure simulation for the 12 screenshot tickers. Orders execute after the signal date, and ATR stops use only prior information.</p>
    <p class='meta'>Massive daily prices, short interest, short volume, NBBO quotes and point-in-time share counts · Initial capital $100,000 · Price returns exclude dividends</p>
    <div class='callout'><strong>Result.</strong> The enhanced strategy ended at ${enhanced['ending_capital']:,.0f}, a {100*enhanced['return']:.1f}% cumulative return across the three non-contiguous tradable windows. Core MA/ATR returned {100*core['return']:.1f}% and passive screenshot-basket exposure returned {100*basket['return']:.1f}%. These are small-event historical results, not evidence that insurance payouts caused the moves.</div>
    <div class='grid'><div class='card'><b>{ATR_MULTIPLE:.1f}× ATR</b>trailing stop</div><div class='card'><b>{best_ticker} {100*best_ret:.1f}%</b>best enhanced ticker</div><div class='card'><b>{worst_ticker} {100*worst_ret:.1f}%</b>worst enhanced ticker</div></div>
    <div class='callout warning'><strong>Alternative-data limit.</strong> Massive short interest begins in December 2017 and daily short volume in February 2024. Those fields cannot be backfilled into 2009–10 or 2015–16. The historical quote input is a single top-of-book NBBO snapshot per week, not a full depth-of-book reconstruction. {future_note}</div>

    <h2>Key findings</h2><ul>
    <li><strong>The enhanced overlay did not improve profit.</strong> It returned {100*enhanced['return']:.2f}% with a {100*enhanced['max_drawdown']:.2f}% maximum drawdown. Core MA/ATR returned {100*core['return']:.2f}% with a {100*core['max_drawdown']:.2f}% drawdown. The extra confirmation reduced risk modestly but cut return sharply.</li>
    <li><strong>Passive exposure dominated in these hand-picked bullish windows.</strong> The screenshot basket gained {100*basket['return']:.1f}%, but suffered a {100*basket['max_drawdown']:.1f}% drawdown. Much of the 2023–24 gain came from highly volatile newer listings, particularly ROOT.</li>
    <li><strong>BAP is the main enhanced-strategy winner.</strong> Its compounded enhanced return was {100*best_ret:.1f}%. BAP is Credicorp, a diversified financial group with an insurance subsidiary, so this cannot be interpreted as a clean catastrophe-insurer payout effect.</li>
    <li><strong>Short/order-book confirmation was weak directionally.</strong> The enhanced portfolio's active-day hit rate was {100*enhanced['active_day_hit']:.1f}%, below a neutral 50% benchmark. BAP and HIG were the most useful confirmations; PGR, KNSL, LMND and ROOT detracted.</li>
    <li><strong>The 2023–24 enhanced overlay returned only 1.00%.</strong> The price-only core returned 8.69% in the same trade window. Short volume was available only from February 2024, so it influenced less than four months of that episode.</li>
    </ul>

    <h2>Strategy comparison</h2>{overall_table}<img class='chart' src='{chart_wealth}' alt='Portfolio wealth comparison'>
    <h2>Why the passive basket can work</h2>
    <p>The passive basket benefits when the event window contains a broad insurance-sector or risk-on repricing: an equal-dollar allocation spreads company-specific risk, keeps exposure through short-lived signal reversals, and lets winners compound without discretionary exits. It also avoids the enhanced strategy's frequent signal changes and ATR-stop losses. This is a mechanism consistent with the results, not proof that El Niño caused the returns.</p>
    <p>Against the <strong>US P&amp;C insurance benchmark (KIE proxy)</strong>, the basket tests whether the predefined company basket added value beyond broad insurance-sector exposure. KIE is an ETF proxy, not a pure index or a claim-loss index; the comparison therefore measures investable sector performance rather than underwriting profitability.</p>
    <p>Returns include {100 * COST_PER_SIDE:.2f}% transaction cost per side at entry and exit and a {100 * CAPITAL_GAINS_TAX_RATE:.0f}% tax on positive realized event gains. The model does not include dividends, which makes this a conservative comparison for dividend-paying insurers.</p>
    <p>The timeline connects only the tested windows. Capital sits in cash between them; the chart removes those blank calendar gaps.</p>

    <h2>Event-level simulation</h2>{event_table}
    <h2>Returns by event window</h2>{event_chart_html}
    <p>Actual severe months are shifted three months to form tradable windows: December 2009–January 2010 becomes March–April 2010; August 2015–March 2016 becomes November 2015–June 2016; September 2023–February 2024 becomes December 2023–May 2024.</p>

    <h2>Rules fixed before testing</h2>
    <ol><li><strong>Peak regime:</strong> ONI ≥ +1.5°C. A three-month delay converts the revised centered ONI observation into an available-information trading flag.</li>
    <li><strong>Trend:</strong> +1 when <code>Close &gt; SMA20 &gt; SMA60</code> and five-day momentum is positive; −1 for the reverse; otherwise zero.</li>
    <li><strong>Volatility filter:</strong> trade only when <code>ATR10 / Close</code> exceeds its trailing 60-session median.</li>
    <li><strong>Order book:</strong> weekly NBBO imbalance <code>I=(bid size−ask size)/(bid size+ask size)</code>. +1 when I&gt;0.10, −1 when I&lt;−0.10. The snapshot is the final quote strictly before 16:00 New York time and becomes usable next session.</li>
    <li><strong>Short interest:</strong> settlement observations are delayed ten calendar days. A decline greater than 5% scores +1; an increase greater than 5% scores −1.</li>
    <li><strong>Short volume:</strong> the five-session short-volume ratio is compared with its trailing 20-session median. A difference beyond ±3 percentage points supplies a weak directional confirmation.</li>
    <li><strong>Composite:</strong> <code>Score=(0.55T+0.20B+0.15SI+0.10SV)/available weights</code>. Trend must be nonzero, agree with the composite, and produce an aligned score of at least 0.55.</li>
    <li><strong>Execution and risk:</strong> next-session open; 2.5× Wilder ATR(10) trailing stop; 7.5 bp per transaction side; 3% annualized short-borrow charge; equal weight across eligible tickers; maximum gross exposure 100%.</li></ol>

    <h2>Enhanced results by ticker and event</h2>{ticker_table}<img class='chart' src='{chart_tickers}' alt='Ticker strategy returns'>

    <h2>Signal quality</h2><p>PnL and signal quality are separate. Direction hit rate tests whether an active close-t signal matched the sign of the next close-to-close return. Score IC is the Pearson correlation between the continuous composite score and next-day return.</p>{quality_table}<img class='chart' src='{chart_quality}' alt='Signal direction hit rate'>

    <h2>Weekly and daily tendencies</h2>{tendency_table}
    <p><strong>Float limitation:</strong> Massive supplies point-in-time share-class and weighted shares outstanding, not free float. “SI/share proxy” divides the most recently available short interest by weighted or share-class shares. It must not be called short interest as a percentage of true tradable float.</p>

    <h2>Alternative-data coverage</h2>{coverage_table}

    <h2>What the result does and does not show</h2><ul>
    <li>The strategy monetizes directional volatility during an ENSO-conditioned window. It does not observe company claim payments, catastrophe reserves, insured-loss announcements or premium repricing directly.</li>
    <li>Short volume includes liquidity provision and hedging. It is not a daily change in short interest and is not automatically bearish.</li>
    <li>Top-of-book imbalance is noisy and can change immediately. A weekly closing snapshot is a coarse directional measure, not an executable historical order book.</li>
    <li>High volatility makes an ATR stop wider in dollars. The 2.5× multiple was fixed, not selected from the winning test result.</li>
    <li>Dividends are excluded. This understates long total returns and overstates short returns. Borrow availability, recalls, locate fees, taxes and market impact are not modeled.</li>
    <li>Only three completed tradable windows exist, and newer listings participate in one. Results are highly vulnerable to event selection, ticker survivorship and one-company news.</li>
    </ul>

    <h2>Metric formulas</h2><p><code>TR_t=max(H_t−L_t, |H_t−C_(t−1)|, |L_t−C_(t−1)|)</code>; ATR10 is Wilder's exponentially smoothed ten-session true range.</p>
    <p><code>Strategy return</code> compounds marked-to-market daily returns after transaction and modeled borrow costs. <code>Ending capital=100,000×∏(1+r_t)</code>.</p>
    <p><code>Maximum drawdown=min_t(W_t/max_(s≤t)W_s−1)</code>. <code>Sharpe=mean(r)/sd(r)×√252</code>, with no risk-free-rate subtraction.</p>
    <p><code>Exposure=mean(|position|)</code>. For the portfolio, this is the mean invested fraction across eligible names.</p>

    <h2>Files and sources</h2><p>Market data use Massive's <a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>daily aggregates</a>, <a href='https://massive.com/docs/rest/stocks/short-interest'>short-interest</a>, <a href='https://massive.com/docs/rest/stocks/short-volume'>short-volume</a>, historical ticker-reference and quote endpoints. ONI comes from the supplied monthly dataset; operational definitions are available from <a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php'>NOAA CPC</a>.</p>
    <p class='small'>Prepared 13 September 2026. Historical research only, not investment advice.</p>
    </main><script>{js}</script></body></html>"""
    REPORT.write_text(report)

    latest_event = (
        event_summary.assign(_trade_end=pd.to_datetime(event_summary["trade_end"]))
        .sort_values("_trade_end")
        .iloc[-1]["episode"]
    )
    latest_summary = event_summary[event_summary["episode"] == latest_event].sort_values("return", ascending=False)
    latest_tickers = ticker_results[ticker_results["episode"] == latest_event].sort_values(["strategy", "return"], ascending=False)
    latest_tendencies = tendencies[tendencies["episode"] == latest_event].sort_values("ticker")
    latest_quality = signal_quality[signal_quality["episode"] == latest_event].sort_values("ticker")
    latest_chart = dict(event_charts)[latest_event]
    latest_start = latest_summary["trade_start"].iloc[0]
    latest_end = latest_summary["trade_end"].iloc[0]
    latest_report = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Latest event strategy report: {html.escape(latest_event)}</title><style>{css}</style></head><body><main>
    <h1>Latest event strategy report</h1>
    <p class='deck'>Detailed results for the latest completed tradable event window: <strong>{html.escape(latest_event)}</strong>.</p>
    <p class='meta'>Trade window: {latest_start} to {latest_end} · Initial capital: $100,000 · Price returns exclude dividends</p>
    <h2>Returns during the event window</h2><img class='chart' src='{latest_chart}' alt='Returns during {html.escape(latest_event)}'>
    <h2>Strategy comparison</h2>{table_html(latest_summary, [("series","Series","text"),("return","Return","pct"),("ending_capital","Ending capital","usd"),("ann_vol","Annualized vol","pct"),("sharpe","Sharpe","num"),("max_drawdown","Max drawdown","pct"),("active_day_hit","Active-day hit","pct"),("exposure","Exposure","pct"),("trades","Trades","int"),("stops","ATR stops","int")], "latest-events")}
    <h2>Results by ticker</h2>{table_html(latest_tickers, [("strategy","Strategy","text"),("ticker","Ticker","text"),("return","Return","pct"),("ending_capital","$100k equivalent","usd"),("ann_vol","Ann. vol","pct"),("sharpe","Sharpe","num"),("max_drawdown","Max drawdown","pct"),("active_day_hit","Active-day hit","pct"),("exposure","Exposure","pct"),("trades","Trades","int"),("stops","Stops","int")], "latest-tickers")}
    <h2>Signal quality</h2>{table_html(latest_quality, [("ticker","Ticker","text"),("observations","Observations","int"),("active_signals","Active signals","int"),("score_next_day_ic","Score IC","num"),("ic_p","IC p-value","num"),("direction_hit_rate","Direction hit","pct"),("mean_signed_next_day","Mean signed next-day","pct")], "latest-quality")}
    <h2>Market and alternative-data tendencies</h2>{table_html(latest_tendencies, [("ticker","Ticker","text"),("annualized_realized_vol","Realized vol","pct"),("mean_atr_pct","Mean ATR/price","pct"),("trend_up_share","Up-trend share","pct"),("trend_down_share","Down-trend share","pct"),("mean_book_imbalance","Mean book imbalance","num"),("median_spread_bps","Median spread (bp)","num"),("mean_short_volume_ratio","Short-volume ratio (%)","num"),("short_interest_pct_shares","SI/share proxy","pct")], "latest-tendencies")}
    <p class='small'>Historical research only, not investment advice. The event chart and tables are generated from the same daily simulation outputs as the full report.</p>
    </main><script>{js}</script></body></html>"""
    LATEST_REPORT.write_text(latest_report)
    print(f"Wrote {REPORT}")
    print(f"Wrote {LATEST_REPORT}")
    print(overall[["series","return","ending_capital","sharpe","max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
