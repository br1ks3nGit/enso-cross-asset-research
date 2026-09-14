#!/usr/bin/env python3
"""Analyze insurer returns against ONI and severe El Nino episodes."""

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
RAW = ROOT / "raw"
PREPARED = ROOT / "prepared"
CALC = ROOT / "calculated"
SOURCE = Path(os.environ.get(
    "INSURANCE_UNIVERSE_CSV",
    str(ROOT.parent / "inputs" / "insurance" / "el_nino_insurance_stocks_massive.csv"),
))
ONI_SOURCE = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(ROOT.parent / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
REPORT = ROOT / "insurance_ONI_event_study.html"
SEVERE_THRESHOLD = 1.5
PUBLICATION_SHIFT_MONTHS = 3


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "n.a." if x is None or pd.isna(x) else f"{100*x:.{digits}f}%"


def fmt_num(x: float | None, digits: int = 3) -> str:
    return "n.a." if x is None or pd.isna(x) else f"{x:.{digits}f}"


def fmt_p(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n.a."
    if x < 0.001:
        return "&lt;0.001"
    return f"{x:.3f}"


def load_raw(ticker: str) -> pd.Series:
    matches = sorted(RAW.glob(f"massive_{ticker}_daily_*.json.gz"))
    if not matches:
        return pd.Series(dtype=float, name=ticker)
    with gzip.open(matches[-1], "rt", encoding="utf-8") as handle:
        rows = json.load(handle).get("results", [])
    if not rows:
        return pd.Series(dtype=float, name=ticker)
    frame = pd.DataFrame(rows)
    idx = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    series = pd.Series(frame["c"].astype(float).to_numpy(), index=idx, name=ticker)
    return series[~series.index.duplicated(keep="last")].sort_index()


def monthly_returns(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    if prices.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    month_close = prices.groupby(prices.index.to_period("M")).last()
    ret = np.log(month_close).diff()
    ordinal = month_close.index.astype(int)
    adjacent = pd.Series(ordinal, index=month_close.index).diff().eq(1)
    ret = ret.where(adjacent)
    return month_close, ret


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(z) < 4 or z["x"].std() == 0 or z["y"].std() == 0:
        return math.nan, math.nan, len(z)
    r, p = stats.pearsonr(z["x"], z["y"])
    return float(r), float(p), len(z)


def hac_slope_test(x: pd.Series, y: pd.Series, lags: int = 6) -> tuple[float, float]:
    """OLS slope and two-sided Newey-West p-value with Bartlett weights."""
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(z) < 8 or z["x"].std() == 0:
        return math.nan, math.nan
    xv = np.column_stack([np.ones(len(z)), z["x"].to_numpy(float)])
    yv = z["y"].to_numpy(float)
    xtxi = np.linalg.inv(xv.T @ xv)
    beta = xtxi @ xv.T @ yv
    resid = yv - xv @ beta
    score = xv * resid[:, None]
    meat = score.T @ score
    for lag in range(1, min(lags, len(z) - 1) + 1):
        weight = 1 - lag / (lags + 1)
        gamma = score[lag:].T @ score[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtxi @ meat @ xtxi * len(z) / max(len(z) - xv.shape[1], 1)
    se = math.sqrt(max(float(cov[1, 1]), 0))
    if se == 0:
        return float(beta[1]), math.nan
    t_stat = float(beta[1] / se)
    p = float(2 * stats.t.sf(abs(t_stat), df=len(z) - 2))
    return float(beta[1]), p


def bh_qvalues(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    m = len(valid)
    if not m:
        return out
    raw = valid.to_numpy(float) * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    out.loc[valid.index] = np.minimum(adjusted, 1.0)
    return out


def severe_episodes(oni: pd.Series) -> list[dict]:
    severe = oni[oni >= SEVERE_THRESHOLD]
    groups: list[list[pd.Period]] = []
    for period in severe.index:
        if not groups or period.ordinal != groups[-1][-1].ordinal + 1:
            groups.append([period])
        else:
            groups[-1].append(period)
    result = []
    for i, periods in enumerate(groups, start=1):
        result.append({
            "episode": f"E{i}: {periods[0]} to {periods[-1]}",
            "start": periods[0],
            "end": periods[-1],
            "months": len(periods),
            "peak_oni": float(oni.loc[periods].max()),
            "partial": periods[-1] == oni.index.max(),
        })
    return result


def price_between(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    before = series[series.index < start]
    through = series[series.index <= end]
    if before.empty or through.empty:
        return math.nan, math.nan
    start_price = float(before.iloc[-1])
    end_price = float(through.iloc[-1])
    return start_price, end_price


def episode_detail(ticker: str, prices: pd.Series, spy: pd.Series, episodes: list[dict]) -> list[dict]:
    if prices.empty:
        return []
    rows: list[dict] = []
    r20 = np.log(prices).diff(20)
    mu20, sd20 = float(r20.mean()), float(r20.std(ddof=1))
    for ep in episodes:
        start = ep["start"].to_timestamp(how="start")
        end = ep["end"].to_timestamp(how="end").normalize()
        window = prices[(prices.index >= start) & (prices.index <= end)]
        p0, p1 = price_between(prices, start, end)
        s0, s1 = price_between(spy, start, end)
        if math.isnan(p0) or window.empty:
            continue
        path = pd.concat([pd.Series([p0], index=[start - pd.Timedelta(days=1)]), window])
        drawdown = path / path.cummax() - 1
        event_r20 = r20[(r20.index >= start) & (r20.index <= end)].dropna()
        best20 = float(event_r20.max()) if not event_r20.empty else math.nan
        worst20 = float(event_r20.min()) if not event_r20.empty else math.nan
        best_date = event_r20.idxmax().date().isoformat() if not event_r20.empty else ""
        worst_date = event_r20.idxmin().date().isoformat() if not event_r20.empty else ""
        best_z = (best20 - mu20) / sd20 if sd20 > 0 and not math.isnan(best20) else math.nan
        worst_z = (worst20 - mu20) / sd20 if sd20 > 0 and not math.isnan(worst20) else math.nan
        spike, plunge = best_z >= 2, worst_z <= -2
        signal = "Mixed" if spike and plunge else "Spike" if spike else "Drawdown" if plunge else "No 2σ move"
        simple_return = p1 / p0 - 1
        spy_return = s1 / s0 - 1 if not math.isnan(s0) else math.nan
        abnormal_log = math.log(p1 / p0) - math.log(s1 / s0) if not math.isnan(s0) else math.nan
        rows.append({
            "ticker": ticker,
            "episode": ep["episode"],
            "start": str(ep["start"]),
            "end": str(ep["end"]),
            "peak_oni": ep["peak_oni"],
            "partial_episode": ep["partial"],
            "trading_days": len(window),
            "episode_return": simple_return,
            "spy_return": spy_return,
            "market_relative_log_return": abnormal_log,
            "max_drawdown": float(drawdown.min()),
            "best_20d_return": math.expm1(best20) if not math.isnan(best20) else math.nan,
            "best_20d_z": best_z,
            "best_20d_end": best_date,
            "worst_20d_return": math.expm1(worst20) if not math.isnan(worst20) else math.nan,
            "worst_20d_z": worst_z,
            "worst_20d_end": worst_date,
            "signal": signal,
        })
    return rows


def image_data(fig) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def html_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], table_id: str, limit: int | None = None) -> str:
    view = frame.head(limit) if limit else frame
    heads = "".join(f"<th onclick=\"sortTable('{table_id}',{i})\">{html.escape(label)}</th>" for i, (_, label, _) in enumerate(columns))
    body = []
    for _, row in view.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row.get(key)
            if kind == "pct":
                shown = fmt_pct(value)
            elif kind == "num":
                shown = fmt_num(value)
            elif kind == "p":
                shown = fmt_p(value)
            elif kind == "int":
                shown = "n.a." if pd.isna(value) else f"{int(value):,}"
            else:
                shown = html.escape(str(value)) if not pd.isna(value) else "n.a."
            cls = " neg" if kind == "pct" and not pd.isna(value) and value < 0 else " pos" if kind == "pct" and not pd.isna(value) and value > 0 else ""
            cells.append(f"<td class='{cls.strip()}'>{shown}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table id='{table_id}'><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def main() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    CALC.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    universe = (
        source[["ticker", "company_name", "insurance_role", "property_insurance",
                "natural_catastrophe_or_reinsurance", "agriculture_insurance"]]
        .drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    )
    metadata = universe.set_index("ticker")

    oni_frame = pd.read_csv(ONI_SOURCE)
    oni_frame["month"] = pd.to_datetime(oni_frame["Date"]).dt.to_period("M")
    oni = oni_frame.set_index("month")["ONI"].astype(float).loc[pd.Period("2001-01", "M"):]
    doni = oni.diff()
    oni_available = oni.shift(PUBLICATION_SHIFT_MONTHS)
    episodes = severe_episodes(oni)

    prices: dict[str, pd.Series] = {ticker: load_raw(ticker) for ticker in universe["ticker"]}
    # Correct ticker histories: pre-2016 CB belongs to another issuer; pre-2023 EG traded as RE.
    ace, re_old = load_raw("ACE"), load_raw("RE")
    prices["CB"] = pd.concat([ace.loc[:"2016-01-14"], prices["CB"].loc["2016-01-15":]]).sort_index()
    prices["EG"] = pd.concat([re_old.loc[:"2023-07-07"], prices["EG"].loc["2023-07-10":]]).sort_index()
    prices["SPY"], prices["KIE"] = load_raw("SPY"), load_raw("KIE")

    monthly_close: dict[str, pd.Series] = {}
    monthly_ret: dict[str, pd.Series] = {}
    for ticker, series in prices.items():
        monthly_close[ticker], monthly_ret[ticker] = monthly_returns(series)

    # Quant-ready long daily and monthly files.
    daily_rows = []
    for ticker in universe["ticker"]:
        for date, close in prices[ticker].items():
            daily_rows.append((date, ticker, float(close)))
    pd.DataFrame(daily_rows, columns=["date", "ticker", "split_adjusted_close"]).to_csv(
        PREPARED / "daily_split_adjusted_prices_long.csv.gz", index=False, compression="gzip"
    )
    monthly_wide = pd.DataFrame(monthly_ret)
    monthly_wide.index = monthly_wide.index.astype(str)
    monthly_wide.index.name = "month"
    monthly_wide.to_csv(PREPARED / "monthly_log_returns_wide.csv.gz", compression="gzip")

    spy_ret = monthly_ret["SPY"]
    severe_mask = oni >= SEVERE_THRESHOLD
    summary_rows = []
    event_rows = []
    for ticker in universe["ticker"]:
        ret = monthly_ret[ticker]
        close = monthly_close[ticker]
        first = prices[ticker].index.min() if not prices[ticker].empty else pd.NaT
        last = prices[ticker].index.max() if not prices[ticker].empty else pd.NaT
        if close.empty:
            coverage_ratio = 0.0
        else:
            expected = max(close.index[-1].ordinal - close.index[0].ordinal + 1, 1)
            coverage_ratio = len(close) / expected
        r_oni, p_oni, n = pearson(oni, ret)
        r_doni, p_doni, _ = pearson(doni, ret)
        r_available, p_available, n_available = pearson(oni_available, ret)
        slope, hac_p = hac_slope_test(oni_available, ret, lags=6)
        abnormal = ret - spy_ret
        r_abnormal, p_abnormal, _ = pearson(oni_available, abnormal)
        joined = pd.concat([ret.rename("ret"), severe_mask.rename("severe")], axis=1).dropna()
        joined["severe"] = joined["severe"].astype(bool)
        severe_returns = joined.loc[joined["severe"], "ret"]
        other_returns = joined.loc[~joined["severe"], "ret"]
        if len(severe_returns) >= 2 and len(other_returns) >= 5:
            severe_p = float(stats.ttest_ind(severe_returns, other_returns, equal_var=False).pvalue)
        else:
            severe_p = math.nan
        _, severe_hac_p = hac_slope_test(joined["severe"].astype(float), joined["ret"], lags=3)
        severe_abnormal = abnormal.reindex(severe_mask[severe_mask].index).dropna()
        severe_n = len(severe_returns)
        if n >= 60 and severe_n >= 6 and coverage_ratio >= 0.80:
            quality = "Usable"
        elif n >= 24 and severe_n >= 2 and coverage_ratio >= 0.50:
            quality = "Limited"
        else:
            quality = "Insufficient"
        summary_rows.append({
            "ticker": ticker,
            "company_name": metadata.loc[ticker, "company_name"],
            "role": metadata.loc[ticker, "insurance_role"],
            "cat_exposed": metadata.loc[ticker, "natural_catastrophe_or_reinsurance"],
            "property_insurance": metadata.loc[ticker, "property_insurance"],
            "first_date": first.date().isoformat() if pd.notna(first) else "",
            "last_date": last.date().isoformat() if pd.notna(last) else "",
            "daily_rows": len(prices[ticker]),
            "monthly_n": n,
            "coverage_ratio": coverage_ratio,
            "severe_months_n": severe_n,
            "quality": quality,
            "corr_oni_r": r_oni,
            "corr_oni_p": p_oni,
            "corr_doni_r": r_doni,
            "corr_doni_p": p_doni,
            "corr_available_r": r_available,
            "corr_available_p": p_available,
            "hac_slope": slope,
            "hac_p": hac_p,
            "corr_market_relative_r": r_abnormal,
            "corr_market_relative_p": p_abnormal,
            "mean_severe_return": float(severe_returns.mean()) if severe_n else math.nan,
            "median_severe_return": float(severe_returns.median()) if severe_n else math.nan,
            "mean_other_return": float(other_returns.mean()) if len(other_returns) else math.nan,
            "severe_minus_other": float(severe_returns.mean() - other_returns.mean()) if severe_n and len(other_returns) else math.nan,
            "severe_welch_p": severe_p,
            "severe_hac_p": severe_hac_p,
            "mean_severe_market_relative": float(severe_abnormal.mean()) if len(severe_abnormal) else math.nan,
        })
        event_rows.extend(episode_detail(ticker, prices[ticker], prices["SPY"], episodes))

    summary = pd.DataFrame(summary_rows)
    eligible = summary["quality"].isin(["Usable", "Limited"])
    summary["hac_q_bh"] = np.nan
    summary.loc[eligible, "hac_q_bh"] = bh_qvalues(summary.loc[eligible, "hac_p"])
    summary["severe_q_bh"] = np.nan
    summary.loc[eligible, "severe_q_bh"] = bh_qvalues(summary.loc[eligible, "severe_hac_p"])
    events = pd.DataFrame(event_rows)
    summary.to_csv(CALC / "company_ONI_statistics.csv", index=False)
    events.to_csv(CALC / "severe_event_company_paths.csv", index=False)

    # Equal-weight catastrophe/property portfolio, fixed each month from available usable names.
    cat_names = summary.loc[
        (summary["quality"] == "Usable")
        & ((summary["cat_exposed"].astype(str).str.lower() == "yes") | (summary["property_insurance"].astype(str).str.lower() == "yes")),
        "ticker",
    ].tolist()
    cat_ret = pd.DataFrame({t: monthly_ret[t] for t in cat_names}).mean(axis=1, skipna=True)
    portfolio_rows = []
    for label, ret in (("Equal-weight catastrophe/property insurers", cat_ret), ("KIE insurance ETF", monthly_ret["KIE"]), ("SPY", spy_ret)):
        r, p, n = pearson(oni_available, ret)
        _, hp = hac_slope_test(oni_available, ret)
        joined = pd.concat([ret.rename("r"), severe_mask.rename("s")], axis=1).dropna()
        joined["s"] = joined["s"].astype(bool)
        severe_r = joined.loc[joined.s, "r"]
        other_r = joined.loc[~joined.s, "r"]
        _, shp = hac_slope_test(joined.s.astype(float), joined.r)
        portfolio_rows.append({
            "series": label, "n": n, "available_oni_r": r, "pearson_p": p, "hac_p": hp,
            "mean_severe_return": severe_r.mean(), "mean_other_return": other_r.mean(),
            "severe_minus_other": severe_r.mean() - other_r.mean(), "severe_hac_p": shp,
        })
    portfolio = pd.DataFrame(portfolio_rows)
    portfolio.to_csv(CALC / "portfolio_benchmark_statistics.csv", index=False)

    # Chart 1: ONI and severe threshold.
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = oni.index.to_timestamp()
    ax.plot(x, oni.values, color="#2563eb", lw=1.7, label="ONI (revised series)")
    ax.axhline(SEVERE_THRESHOLD, color="#b91c1c", lw=1, ls="--", label="Severe threshold (+1.5°C)")
    ax.fill_between(x, SEVERE_THRESHOLD, oni.values, where=oni.values >= SEVERE_THRESHOLD, color="#f97316", alpha=.3)
    ax.axhline(0, color="#64748b", lw=.7)
    ax.set(title="ONI and severe El Niño months", ylabel="ONI (°C)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.2)
    chart_oni = image_data(fig)

    # Chart 2: correlations among usable names.
    corr_plot = summary[summary["quality"] == "Usable"].sort_values("corr_available_r")
    show = pd.concat([corr_plot.head(12), corr_plot.tail(12)]).drop_duplicates("ticker").sort_values("corr_available_r")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(show["hac_q_bh"] < .10, "#b91c1c", "#2563eb")
    ax.barh(show["ticker"], show["corr_available_r"], color=colors)
    ax.axvline(0, color="#475569", lw=.8)
    ax.set(title="Largest positive and negative raw correlations\navailable ONI vs monthly stock return", xlabel="Pearson r")
    ax.grid(axis="x", alpha=.2)
    chart_corr = image_data(fig)

    # Chart 3: most extreme 20-day price moves during severe episodes.
    usable_tickers = set(summary.loc[summary["quality"] == "Usable", "ticker"])
    extremes = events[events["ticker"].isin(usable_tickers)].copy()
    extremes["extreme_z"] = np.where(extremes["best_20d_z"].abs() >= extremes["worst_20d_z"].abs(), extremes["best_20d_z"], extremes["worst_20d_z"])
    extremes["label"] = extremes["ticker"] + " · " + extremes["start"]
    extremes_plot = extremes.loc[extremes["extreme_z"].abs().nlargest(18).index].sort_values("extreme_z")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(extremes_plot["label"], extremes_plot["extreme_z"], color=np.where(extremes_plot["extreme_z"] < 0, "#b91c1c", "#15803d"))
    ax.axvline(-2, color="#b91c1c", lw=.8, ls="--")
    ax.axvline(2, color="#15803d", lw=.8, ls="--")
    ax.set(title="Most unusual 20-trading-day moves inside severe episodes", xlabel="Z-score vs each stock's full-history 20-day returns")
    ax.grid(axis="x", alpha=.2)
    chart_extreme = image_data(fig)

    usable = summary[summary["quality"] == "Usable"].copy()
    significant_corr = usable[usable["hac_q_bh"] < .10].sort_values("hac_q_bh")
    significant_event = usable[usable["severe_q_bh"] < .10].sort_values("severe_q_bh")
    strongest = usable.reindex(usable["corr_available_r"].abs().sort_values(ascending=False).index).head(12)
    event_rank = usable.sort_values("mean_severe_market_relative").head(12)
    flagged = extremes[extremes["signal"] != "No 2σ move"].copy()
    flagged["abs_z"] = flagged[["best_20d_z", "worst_20d_z"]].abs().max(axis=1)
    flagged = flagged.sort_values("abs_z", ascending=False)

    episode_text = ", ".join(
        f"{e['start']}–{e['end']} (peak {e['peak_oni']:.1f}°C{' partial' if e['partial'] else ''})" for e in episodes
    )
    robust_corr_text = (
        f"{len(significant_corr)} companies retain q&lt;0.10 after multiple-testing correction."
        if len(significant_corr)
        else "No company retains q&lt;0.10 after correcting the cross-section for multiple tests."
    )
    robust_event_text = (
        f"{len(significant_event)} companies show a severe-month return difference with q&lt;0.10."
        if len(significant_event)
        else "No company shows a severe-month return difference with q&lt;0.10 after multiple-testing correction."
    )
    root_row = summary.set_index("ticker").loc["ROOT"]
    plmr_row = summary.set_index("ticker").loc["PLMR"]
    cat_row = portfolio.set_index("series").loc["Equal-weight catastrophe/property insurers"]
    uve_event = events[(events["ticker"] == "UVE") & (events["start"] == "2015-08")].iloc[0]

    summary_table = html_table(
        strongest,
        [("ticker", "Ticker", "text"), ("company_name", "Company", "text"), ("corr_available_r", "Raw r", "num"),
         ("hac_p", "HAC p", "p"), ("hac_q_bh", "BH q", "p"), ("mean_severe_return", "Mean severe-month return", "pct"),
         ("mean_severe_market_relative", "Mean vs SPY", "pct"), ("quality", "Data", "text")],
        "corrTable",
    )
    event_table = html_table(
        flagged,
        [("ticker", "Ticker", "text"), ("episode", "Episode", "text"), ("signal", "Signal", "text"),
         ("episode_return", "Episode return", "pct"), ("market_relative_log_return", "Log return vs SPY", "pct"),
         ("max_drawdown", "Max drawdown", "pct"), ("best_20d_return", "Best 20d", "pct"),
         ("best_20d_z", "Best z", "num"), ("worst_20d_return", "Worst 20d", "pct"), ("worst_20d_z", "Worst z", "num")],
        "eventTable", limit=40,
    )
    all_table = html_table(
        summary.sort_values(["quality", "corr_available_r"], ascending=[True, True]),
        [("ticker", "Ticker", "text"), ("company_name", "Company", "text"), ("first_date", "Start", "text"),
         ("last_date", "End", "text"), ("monthly_n", "Months", "int"), ("coverage_ratio", "Coverage", "pct"),
         ("severe_months_n", "Severe months", "int"), ("corr_oni_r", "Contemp. r", "num"),
         ("corr_oni_p", "Contemp. p", "p"), ("corr_available_r", "Available-ONI r", "num"),
         ("hac_p", "HAC p", "p"), ("hac_q_bh", "BH q", "p"), ("severe_minus_other", "Severe − other", "pct"),
         ("severe_hac_p", "Event HAC p", "p"), ("severe_q_bh", "Event BH q", "p"), ("quality", "Data", "text")],
        "allTable",
    )
    portfolio_table = html_table(
        portfolio,
        [("series", "Series", "text"), ("n", "Months", "int"), ("available_oni_r", "Raw r", "num"),
         ("hac_p", "HAC p", "p"), ("mean_severe_return", "Mean severe return", "pct"),
         ("mean_other_return", "Mean other return", "pct"), ("severe_minus_other", "Difference", "pct"),
         ("severe_hac_p", "Event HAC p", "p")],
        "portfolioTable",
    )

    css = """
    :root{--ink:#172033;--muted:#5f6b7a;--line:#dbe2ea;--blue:#1d4ed8;--navy:#0f2747;--red:#b91c1c;--green:#15803d;--paper:#fff;--wash:#f6f8fb}
    *{box-sizing:border-box} body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.52 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
    main{max-width:1180px;margin:32px auto;background:var(--paper);padding:48px 54px 72px;box-shadow:0 8px 30px #20304018}
    h1{font-size:34px;line-height:1.12;margin:0 0 10px;color:var(--navy)} h2{font-size:23px;margin:42px 0 12px;color:var(--navy);border-top:1px solid var(--line);padding-top:28px} h3{font-size:17px;margin:26px 0 8px}
    .deck{font-size:18px;color:var(--muted);max-width:850px}.meta{color:var(--muted);font-size:13px;margin:14px 0 26px}.callout{border-left:5px solid var(--blue);background:#eff6ff;padding:18px 20px;margin:22px 0}.warning{border-left-color:#d97706;background:#fff7ed}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.card{border:1px solid var(--line);padding:17px;border-radius:8px}.card b{display:block;font-size:25px;color:var(--navy)}
    img.chart{display:block;max-width:100%;margin:18px auto 8px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px;margin:14px 0 22px}table{border-collapse:collapse;width:100%;font-size:12.5px}th{background:var(--navy);color:#fff;padding:9px 8px;white-space:nowrap;cursor:pointer;position:sticky;top:0}td{padding:8px;border-bottom:1px solid #e8edf3;white-space:nowrap;text-align:right}td:nth-child(1),td:nth-child(2),td:nth-child(3){text-align:left}tr:nth-child(even){background:#f8fafc}.neg{color:var(--red)}.pos{color:var(--green)}code{background:#eef2f7;padding:2px 5px;border-radius:4px}li{margin:7px 0}.small{font-size:13px;color:var(--muted)}a{color:var(--blue)}
    @media(max-width:760px){main{margin:0;padding:28px 20px}.grid{grid-template-columns:1fr}h1{font-size:28px}}
    """
    js = """
    function sortTable(id,n){const t=document.getElementById(id),b=t.tBodies[0],r=[...b.rows];const asc=t.dataset.sortcol==n&&t.dataset.dir!='asc';r.sort((a,c)=>{let x=a.cells[n].innerText.replace(/[%,<>]/g,''),y=c.cells[n].innerText.replace(/[%,<>]/g,'');let nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;return asc?x.localeCompare(y):y.localeCompare(x)});r.forEach(x=>b.appendChild(x));t.dataset.sortcol=n;t.dataset.dir=asc?'asc':'desc'}
    """
    report = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Insurance stocks and severe El Niño</title><style>{css}</style></head><body><main>
    <h1>Insurance stocks and severe El Niño</h1>
    <p class='deck'>Daily Massive prices, monthly ONI correlation, and severe-event price reactions. Study window is constrained by actual vendor and listing coverage, not by the requested date label.</p>
    <p class='meta'>Prepared 13 September 2026 · Requested 2001–2026 · ONI observations through July 2026 · Split-adjusted price returns, excluding dividends</p>
    <div class='callout'><strong>Bottom line.</strong> {robust_corr_text} {robust_event_text} Raw magnitudes can rank sensitivity, but the small number of severe episodes and survivorship-biased company list do not support a causal catastrophe-loss trading claim.</div>
    <div class='grid'><div class='card'><b>{len(universe)}</b>companies requested</div><div class='card'><b>{int((summary.quality=='Usable').sum())}</b>usable histories</div><div class='card'><b>{len(episodes)}</b>severe episodes, one partial</div></div>

    <h2>Key findings</h2>
    <ul>
      <li><strong>Two names pass the cross-sectional screen, but neither has multi-event history.</strong> ROOT and PLMR retain q&lt;0.10 for available-ONI correlation. Both are young listings with only seven severe months in the sample, all concentrated in one completed episode plus July 2026.</li>
      <li><strong>The largest raw correlations are concentrated in young listings.</strong> ROOT has r={root_row['corr_available_r']:.3f}, HAC p={root_row['hac_p']:.3f}, q={root_row['hac_q_bh']:.3f}; PLMR has r={plmr_row['corr_available_r']:.3f}, HAC p={plmr_row['hac_p']:.3f}, q={plmr_row['hac_q_bh']:.3f}. Each overlaps only one completed severe episode, so these are fragile ranks rather than stable signals.</li>
      <li><strong>The catastrophe/property basket did not sell off in severe months.</strong> It averaged {100*cat_row['mean_severe_return']:.2f}% in severe months versus {100*cat_row['mean_other_return']:.2f}% otherwise, a {100*cat_row['severe_minus_other']:.2f}-point difference with HAC p={cat_row['severe_hac_p']:.3f}. The sign is positive but statistically inconclusive.</li>
      <li><strong>The clearest drawdown candidate is company-specific, not proof of an ENSO channel.</strong> UVE lost {100*uve_event['episode_return']:.1f}% over the 2015-08 to 2016-03 severe window, had a {100*uve_event['max_drawdown']:.1f}% maximum drawdown, and lagged SPY by {100*uve_event['market_relative_log_return']:.1f}% on a log-return basis.</li>
      <li><strong>Extreme moves often reverse inside the same episode.</strong> A “Mixed” flag means both a +2σ spike and a −2σ drawdown occurred; this is volatility, not a directional loss signal.</li>
    </ul>

    <h2>What was tested</h2>
    <p>The analysis uses monthly log price returns, not price levels. Price levels trend and would create spurious correlation with a persistent climate index. A severe month is defined here as ONI ≥ +1.5°C. The episodes in the supplied ONI file are {episode_text}.</p>
    <p>Two alignments are shown. <strong>Contemporaneous ONI</strong> matches the revised ONI value to the same month's return and is descriptive only. <strong>Available ONI</strong> shifts ONI by three months, a conservative publication-timing convention for a centered three-month index. It prevents a trading interpretation from receiving the current revised value prematurely; it is still not a true NOAA vintage archive.</p>
    <img class='chart' src='{chart_oni}' alt='ONI series and severe threshold'>

    <h2>Main correlation result</h2>
    <p>The table ranks the largest absolute raw Pearson correlations among usable companies. <strong>Raw r</strong> is direction and magnitude only. HAC p-values allow six months of autocorrelation; BH q-values correct for testing many companies. A large absolute r paired with a large p or q is not reliable evidence.</p>
    {summary_table}
    <img class='chart' src='{chart_corr}' alt='Largest raw correlations'>

    <h2>Severe-event spikes and drawdowns</h2>
    <p>A price move is flagged when its best or worst 20-trading-day log return inside a severe episode is at least two full-history standard deviations from that stock's usual 20-day return. This detects unusual moves; it does not attribute them to insured catastrophe losses. Episode return is measured from the last close before the first severe month through the last close in the final severe month.</p>
    {event_table if len(flagged) else '<p>No usable company produced a ±2σ 20-day move inside the severe windows.</p>'}
    <img class='chart' src='{chart_extreme}' alt='Extreme event-window price moves'>

    <h2>Portfolio and benchmarks</h2>
    <p>The equal-weight portfolio averages available monthly returns of usable property/catastrophe-exposed companies. KIE is the insurance-sector benchmark and SPY is the broad-market benchmark. This cross-check asks whether a company result is distinct from the sector or market.</p>
    {portfolio_table}

    <h2>All companies and data quality</h2>
    <p>Massive returned at least one bar for 58 of the 60 supplied tickers. ODMTY and SNTAY returned no daily aggregates. Several OTC listings have only post-2021 or sporadic prints and are marked insufficient. CB is spliced from ACE through 14 January 2016 and CB thereafter; EG is spliced from RE through 7 July 2023 and EG thereafter.</p>
    {all_table}

    <h2>Interpretation</h2>
    <ul>
      <li><strong>Insurance premiums are not losses.</strong> Premiums are revenue. Catastrophe claims and loss-adjustment expenses reduce underwriting profit; later premium repricing can improve future margins.</li>
      <li><strong>ONI is a broad climate-state proxy.</strong> It does not identify the peril, landfall, insured geography, policy limits, reinsurance attachment, or claims booked by a specific company.</li>
      <li><strong>Stock returns mix many channels.</strong> Catastrophe claims, reserve revisions, reinsurance recoveries, premium pricing, investment income, interest rates and market beta can offset one another.</li>
      <li><strong>Statistical significance is not economic causality.</strong> A p-value is the probability of data at least this extreme under a specified zero-effect model. It is not the probability that the hypothesis is true. BH q-values control the expected false-discovery share among selected findings.</li>
      <li><strong>The event count is small.</strong> The threshold produces three completed severe episodes after broad daily stock coverage begins, plus a one-month partial 2026 episode. Monthly sample size therefore exaggerates apparent precision if clustering is ignored.</li>
      <li><strong>Survivorship bias is material.</strong> The supplied list contains securities observable in 2026. Failed, acquired or delisted insurers are absent, so average historical resilience can be overstated.</li>
    </ul>

    <h2>Metric definitions</h2>
    <p><code>Monthly log return = ln(P_t / P_(t−1))</code>, using the last split-adjusted daily close in each adjacent calendar month. Dividends are excluded.</p>
    <p><code>Pearson r = cov(X,Y) / (sd(X)·sd(Y))</code>. Here X is ONI and Y is monthly stock return. Values range from −1 to +1.</p>
    <p><code>HAC p</code> is the two-sided p-value for the ONI slope in <code>return_t = α + β·ONI_available,t + ε_t</code>, with Newey–West standard errors and six monthly lags.</p>
    <p><code>Severe − other = mean(return | ONI≥1.5) − mean(return | ONI&lt;1.5)</code>. Event HAC p tests the severe-month indicator using the same autocorrelation correction.</p>
    <p><code>Market-relative log return = ln(P_1/P_0) − ln(SPY_1/SPY_0)</code>. This is not a regression alpha.</p>
    <p><code>Maximum drawdown = min_t(P_t / max_(s≤t) P_s − 1)</code> within each severe episode, anchored to the last close before onset.</p>
    <p><code>20-day z = (20-day log return − full-history mean) / full-history standard deviation</code>.</p>

    <h2>Data and reproducibility</h2>
    <p>Prices come from the <a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>Massive custom-bars endpoint</a>, requested at daily granularity with <code>adjusted=true</code>. The ONI series is the supplied monthly file; NOAA's operational definitions and current table are available from the <a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php'>Climate Prediction Center</a>. Raw Massive responses, hashes and the exact download range are retained locally.</p>
    <p class='small'>Files: prepared daily prices, monthly returns, download manifest, company statistics, episode-level paths and portfolio statistics. This is historical research, not investment advice.</p>
    </main><script>{js}</script></body></html>"""
    REPORT.write_text(report)
    print(f"Wrote {REPORT}")
    print(json.dumps({
        "companies": len(summary), "usable": int((summary.quality == "Usable").sum()),
        "limited": int((summary.quality == "Limited").sum()), "insufficient": int((summary.quality == "Insufficient").sum()),
        "significant_hac_q10": len(significant_corr), "significant_event_q10": len(significant_event),
        "flagged_episode_moves": len(flagged), "cat_portfolio_names": len(cat_names),
    }, indent=2))


if __name__ == "__main__":
    main()
