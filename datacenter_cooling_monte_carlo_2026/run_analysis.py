#!/usr/bin/env python3
"""Scenario-conditioned Monte Carlo for listed data-center cooling suppliers."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CALC = ROOT / "calculated"
REPORT = ROOT / "datacenter_cooling_stock_monte_carlo_oct_dec_2026.html"
PRIOR = ROOT.parent
METALS_PATH = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(PRIOR / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
FORECAST_PATH = PRIOR / "enso_forecast_2026/calculated/oni_forecast.json"
STATE_EFFECT_PATH = PRIOR / "enso_forecast_2026/calculated/state_enso_effects.json"
STATE_TEMP_PATH = PRIOR / "enso_forecast_2026/calculated/state_temperature.jsonl.gz"
SUPPLIER_PATH = PRIOR / "enso_datacenter_cooling_capex_2026/calculated/public_supplier_screen.csv"

TICKERS = ["MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC"]
NAMES = {
    "MOD": "Modine", "AAON": "AAON", "VRT": "Vertiv", "NVT": "nVent",
    "JCI": "Johnson Controls", "TT": "Trane Technologies", "SPXC": "SPX Technologies",
}
STARTS = {
    "MOD": "2004-11", "AAON": "2004-02", "VRT": "2020-03", "NVT": "2018-05",
    "JCI": "2016-10", "TT": "2020-04", "SPXC": "2015-10",
}
CAPACITY_GW = {"Illinois": 1.746, "Indiana": 0.061, "Iowa": 0.660,
               "Michigan": 0.127, "North Dakota": 1.090, "Ohio": 0.892}
FEATURES = ["XLI", "Copper", "Aluminum", "Nickel", "ONI", "dONI", "MidwestTemp"]
N_PATHS = 10_000
SEED = 20260913
TRAIN_END = pd.Period("2026-07", "M")
MONTHS = ["Oct 2026", "Nov 2026", "Dec 2026"]


def load_monthly(path: Path) -> pd.Series:
    rows = json.loads(path.read_text())["results"]
    return pd.Series(
        {pd.to_datetime(r["t"], unit="ms").to_period("M"): float(r["c"]) for r in rows}
    ).sort_index()


def temperature_index() -> pd.Series:
    data = pd.read_json(STATE_TEMP_PATH, lines=True)
    data = data[data["state"].isin(CAPACITY_GW)].copy()
    normals = (
        data[data["year"].between(1991, 2020)]
        .groupby(["state", "month"])["temperature_c"].mean().rename("normal")
    )
    data = data.join(normals, on=["state", "month"])
    data["anomaly"] = data["temperature_c"] - data["normal"]
    data["weight"] = data["state"].map(CAPACITY_GW)
    data["weighted"] = data["anomaly"] * data["weight"]
    out = data.groupby(["year", "month"]).agg(weighted=("weighted", "sum"), weight=("weight", "sum"))
    out["value"] = out["weighted"] / out["weight"]
    return pd.Series(
        out["value"].to_numpy(),
        index=pd.PeriodIndex(
            [f"{year:04d}-{month:02d}" for year, month in out.index], freq="M"
        ),
        name="MidwestTemp",
    ).sort_index()


def load_factors_and_returns() -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, float]]:
    base = pd.read_csv(METALS_PATH)
    base.index = pd.to_datetime(base["Date"]).dt.to_period("M")
    factors = pd.DataFrame(index=base.index)
    factors["ONI"] = base["ONI"].to_numpy(float)
    factors["dONI"] = factors["ONI"].diff()
    for metal in ["Copper", "Aluminum", "Nickel"]:
        factors[metal] = np.log(base[metal].astype(float)).diff().to_numpy()
    xli = load_monthly(RAW / "massive_XLI_monthly.json")
    factors["XLI"] = np.log(xli).diff()
    factors["MidwestTemp"] = temperature_index()
    factors = factors.loc[:TRAIN_END].replace([np.inf, -np.inf], np.nan)

    stock_returns: dict[str, pd.Series] = {}
    last_prices: dict[str, float] = {}
    for ticker in TICKERS:
        price = load_monthly(RAW / f"massive_{ticker}_monthly.json")
        last_prices[ticker] = float(price.iloc[-1])
        stock_returns[ticker] = np.log(price).diff().loc[pd.Period(STARTS[ticker], "M"):TRAIN_END]
    return factors, stock_returns, last_prices


def fit_ridge(x: pd.DataFrame, y: pd.Series) -> dict:
    z = x.join(y.rename("y"), how="inner").dropna()
    X, Y = z[FEATURES].to_numpy(float), z["y"].to_numpy(float)
    n_splits = 5 if len(z) >= 60 else 4
    splitter = TimeSeriesSplit(n_splits=n_splits)
    alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    cv_rows = []
    for alpha in alphas:
        scores = []
        for train, test in splitter.split(X):
            scaler = StandardScaler().fit(X[train])
            train_mean = float(Y[train].mean())
            model = Ridge(alpha=alpha, fit_intercept=False).fit(scaler.transform(X[train]), Y[train] - train_mean)
            scores.append(mean_squared_error(Y[test], train_mean + model.predict(scaler.transform(X[test]))))
        cv_rows.append((float(np.mean(scores)), alpha))
    _, alpha = min(cv_rows)

    # Blocked out-of-sample predictions at the chosen penalty.
    oos_true, oos_pred = [], []
    for train, test in splitter.split(X):
        scaler_fold = StandardScaler().fit(X[train])
        train_mean = float(Y[train].mean())
        model_fold = Ridge(alpha=alpha, fit_intercept=False).fit(scaler_fold.transform(X[train]), Y[train] - train_mean)
        oos_true.extend(Y[test])
        oos_pred.extend(train_mean + model_fold.predict(scaler_fold.transform(X[test])))

    scaler = StandardScaler().fit(X)
    historical_mean = float(Y.mean())
    model = Ridge(alpha=alpha, fit_intercept=False).fit(scaler.transform(X), Y - historical_mean)
    fitted = historical_mean + model.predict(scaler.transform(X))
    residual = pd.Series(Y - fitted, index=z.index)
    return {
        "frame": z, "scaler": scaler, "model": model, "residual": residual,
        "alpha": alpha, "n": len(z), "start": str(z.index[0]), "end": str(z.index[-1]),
        "historical_mean": historical_mean,
        "in_r2": float(r2_score(Y, fitted)),
        "oos_r2": float(r2_score(oos_true, oos_pred)),
        "oos_corr": float(np.corrcoef(oos_true, oos_pred)[0, 1]),
        "resid_vol": float(residual.std(ddof=1) * math.sqrt(12)),
    }


def weighted_state_paths() -> dict[str, np.ndarray]:
    estimates = pd.DataFrame(json.loads(STATE_EFFECT_PATH.read_text())["estimates"])
    estimates = estimates[estimates["state"].isin(CAPACITY_GW) & estimates["month"].isin([10, 11, 12])].copy()
    estimates["weight"] = estimates["state"].map(CAPACITY_GW)
    out = {}
    for label, col in [("Low", "p10_c"), ("Central", "enso_effect_c"), ("High", "p90_c")]:
        vals = []
        for month in [10, 11, 12]:
            g = estimates[estimates["month"] == month]
            vals.append(float(np.average(g[col], weights=g["weight"])))
        out[label] = np.array(vals)
    return out


def oni_paths() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    fc = pd.DataFrame(json.loads(FORECAST_PATH.read_text())["forecast"]).set_index("month")
    paths = {
        "Low": fc.loc[[10, 11, 12], "p10_c"].to_numpy(float),
        "Central": fc.loc[[10, 11, 12], "oni_c"].to_numpy(float),
        "High": fc.loc[[10, 11, 12], "p90_c"].to_numpy(float),
    }
    sep = {"Low": float(fc.loc[9, "p10_c"]), "Central": float(fc.loc[9, "oni_c"]), "High": float(fc.loc[9, "p90_c"])}
    return paths, sep


def ar1_metal_means(factors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    means, sds, residuals = [], [], []
    cols = ["Copper", "Aluminum", "Nickel"]
    for col in cols:
        r = factors[col].dropna()
        x, y = r.iloc[:-1].to_numpy(), r.iloc[1:].to_numpy()
        phi, intercept = np.polyfit(x, y, 1)
        phi = float(np.clip(phi, -0.5, 0.5))
        intercept = float(y.mean() - phi * x.mean())
        current = float(r.iloc[-1])
        path = []
        for _ in range(3):
            current = intercept + phi * current
            path.append(current)
        means.append(path)
        innovation = y - (intercept + phi * x)
        residuals.append(pd.Series(innovation, index=r.index[1:], name=col))
        sds.append(float(r.std(ddof=1)))
    return np.array(means).T, np.array(sds), pd.concat(residuals, axis=1).dropna()


def metal_scenario_frame(factors: pd.DataFrame) -> pd.DataFrame:
    base, sd, _ = ar1_metal_means(factors)
    rows = []
    for name, path in [("Down", base - 0.50 * sd), ("Base", base), ("Up", base + 0.50 * sd)]:
        for i, month in enumerate(MONTHS):
            rows.append({"scenario": name, "month": month, "Copper": path[i, 0],
                         "Aluminum": path[i, 1], "Nickel": path[i, 2]})
    return pd.DataFrame(rows)


def sample_blocks(values: np.ndarray, n_paths: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular empirical moving-block bootstrap: output (paths, length, columns)."""
    if values.ndim == 1:
        values = values[:, None]
    starts = rng.integers(0, len(values), size=n_paths)
    idx = (starts[:, None] + np.arange(length)[None, :]) % len(values)
    return values[idx]


def metrics_from_paths(paths: np.ndarray, start_price: float) -> dict[str, float]:
    end_return = paths[:, -1] / start_price - 1
    p05 = float(np.quantile(end_return, 0.05))
    return {
        "start_price": start_price,
        "mean_return": float(end_return.mean()), "median_return": float(np.median(end_return)),
        "p05_return": p05, "p25_return": float(np.quantile(end_return, 0.25)),
        "p75_return": float(np.quantile(end_return, 0.75)), "p95_return": float(np.quantile(end_return, 0.95)),
        "prob_positive": float((end_return > 0).mean()),
        "prob_gain20": float((end_return > 0.20).mean()), "prob_loss20": float((end_return < -0.20).mean()),
        "cvar05_return": float(end_return[end_return <= p05].mean()),
        "median_end_price": float(np.median(paths[:, -1])),
        "p05_end_price": float(np.quantile(paths[:, -1], 0.05)), "p95_end_price": float(np.quantile(paths[:, -1], 0.95)),
    }


def run_simulation(factors: pd.DataFrame, models: dict, last_prices: dict[str, float]) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    oni, sep_oni = oni_paths()
    temp = weighted_state_paths()
    metal_base, metal_sd, metal_residuals = ar1_metal_means(factors)
    metal_means = {
        "Down": metal_base - 0.50 * metal_sd,
        "Base": metal_base,
        "Up": metal_base + 0.50 * metal_sd,
    }

    # Common random numbers make scenario comparisons stable.
    market_hist = factors["XLI"].dropna().to_numpy(float)
    market_draw = sample_blocks(market_hist, N_PATHS, 3, rng)[:, :, 0]
    metal_draw = sample_blocks(metal_residuals.to_numpy(float), N_PATHS, 3, rng)

    common_resid = pd.concat({t: models[t]["residual"] for t in TICKERS}, axis=1).dropna()
    common_resid = common_resid - common_resid.mean()
    resid_draw = sample_blocks(common_resid.to_numpy(float), N_PATHS, 3, rng)

    hist_oni_min, hist_oni_max = factors["ONI"].min(), factors["ONI"].max()
    doni_low, doni_high = factors["dONI"].quantile([0.01, 0.99])
    rows, stored, fan_rows = [], {}, []
    for oni_name, oni_raw in oni.items():
        doni_raw = np.diff(np.r_[sep_oni[oni_name], oni_raw])
        oni_model = np.clip(oni_raw, hist_oni_min, hist_oni_max)
        doni_model = np.clip(doni_raw, doni_low, doni_high)
        for temp_name, temp_path in temp.items():
            for metal_name, metal_mean in metal_means.items():
                scenario = f"ONI {oni_name} | Temp {temp_name} | Metals {metal_name}"
                metal_paths = metal_mean[None, :, :] + metal_draw
                stock_paths = {}
                for j, ticker in enumerate(TICKERS):
                    X = np.empty((N_PATHS, 3, len(FEATURES)))
                    X[:, :, 0] = market_draw
                    X[:, :, 1:4] = metal_paths
                    X[:, :, 4] = oni_model[None, :]
                    X[:, :, 5] = doni_model[None, :]
                    X[:, :, 6] = temp_path[None, :]
                    flat = X.reshape(-1, len(FEATURES))
                    m = models[ticker]
                    predicted = m["model"].predict(m["scaler"].transform(flat)).reshape(N_PATHS, 3)
                    log_returns = predicted + resid_draw[:, :, j]
                    price_paths = last_prices[ticker] * np.exp(np.cumsum(log_returns, axis=1))
                    stock_paths[ticker] = price_paths
                    row = {"scenario": scenario, "oni": oni_name, "temperature": temp_name, "metals": metal_name,
                           "ticker": ticker, **metrics_from_paths(price_paths, last_prices[ticker])}
                    rows.append(row)
                basket = np.mean(np.stack([stock_paths[t] / last_prices[t] * 100 for t in TICKERS], axis=2), axis=2)
                rows.append({"scenario": scenario, "oni": oni_name, "temperature": temp_name, "metals": metal_name,
                             "ticker": "Basket", **metrics_from_paths(basket, 100.0)})
                stored[scenario] = {**stock_paths, "Basket": basket, "market": market_draw}
                if oni_name == "Central" and temp_name == "Central" and metal_name == "Base":
                    for ticker, paths in {**stock_paths, "Basket": basket}.items():
                        start = last_prices.get(ticker, 100.0)
                        for k, month in enumerate(MONTHS):
                            fan_rows.append({
                                "ticker": ticker, "month": month, "p05": np.quantile(paths[:, k], .05),
                                "p25": np.quantile(paths[:, k], .25), "median": np.quantile(paths[:, k], .5),
                                "p75": np.quantile(paths[:, k], .75), "p95": np.quantile(paths[:, k], .95),
                                "start_price": start,
                            })

    results = pd.DataFrame(rows)

    # Market-regime diagnostics within the central climate/metals scenario.
    central_key = "ONI Central | Temp Central | Metals Base"
    central = stored[central_key]
    mkt_cum = np.exp(central["market"].sum(axis=1)) - 1
    regime = np.select([mkt_cum < -0.05, mkt_cum > 0.05], ["Risk-off: XLI < -5%", "Risk-on: XLI > +5%"], default="Sideways: -5% to +5%")
    regime_rows = []
    for ticker in TICKERS + ["Basket"]:
        start = last_prices.get(ticker, 100.0)
        end_ret = central[ticker][:, -1] / start - 1
        for label in ["Risk-off: XLI < -5%", "Sideways: -5% to +5%", "Risk-on: XLI > +5%"]:
            mask = regime == label
            regime_rows.append({"ticker": ticker, "regime": label, "n": int(mask.sum()),
                                "median_return": float(np.median(end_ret[mask])),
                                "prob_positive": float((end_ret[mask] > 0).mean())})
    return results, stored, pd.DataFrame(fan_rows), pd.DataFrame(regime_rows)


def coefficient_table(models: dict) -> pd.DataFrame:
    rows = []
    for ticker, result in models.items():
        coef = dict(zip(FEATURES, result["model"].coef_))
        rows.append({"ticker": ticker, "alpha": result["alpha"], "n": result["n"], "start": result["start"],
                     "historical_mean": result["historical_mean"], "in_r2": result["in_r2"], "oos_r2": result["oos_r2"], "oos_corr": result["oos_corr"],
                     "resid_vol": result["resid_vol"], **coef})
    return pd.DataFrame(rows)


def fmt_pct(x, digits=1) -> str:
    return "n.a." if pd.isna(x) else f"{100 * float(x):.{digits}f}%"


def fmt_num(x, digits=2) -> str:
    return "n.a." if pd.isna(x) else f"{float(x):,.{digits}f}"


def table(frame: pd.DataFrame, columns: list[tuple[str, str, object]], cls="") -> str:
    heads = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row.get(key)
            rendered = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap {cls}'><table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def fan_svg(fan: pd.DataFrame, ticker="Basket", width=920, height=390) -> str:
    d = fan[fan["ticker"] == ticker].copy()
    start = float(d["start_price"].iloc[0])
    points = [{"month": "Sep 11", "p05": start, "p25": start, "median": start, "p75": start, "p95": start}]
    points += d[["month", "p05", "p25", "median", "p75", "p95"]].to_dict("records")
    lo, hi = min(p["p05"] for p in points), max(p["p95"] for p in points)
    pad = max((hi - lo) * .12, 1)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 65, 28, 28, 52
    def xy(i, v):
        x = left + i * (width - left - right) / (len(points) - 1)
        y = top + (hi - v) * (height - top - bottom) / (hi - lo)
        return x, y
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{ticker} Monte Carlo fan chart'>"]
    for frac in np.linspace(0, 1, 5):
        val = lo + frac * (hi - lo); y = xy(0, val)[1]
        parts.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#d8dedf'/><text x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{val:.0f}</text>")
    for low, high, color in [("p05", "p95", "#cfe6e2"), ("p25", "p75", "#8fc7bd")]:
        upper = " ".join(f"{xy(i,p[high])[0]:.1f},{xy(i,p[high])[1]:.1f}" for i,p in enumerate(points))
        lower = " ".join(f"{xy(i,p[low])[0]:.1f},{xy(i,p[low])[1]:.1f}" for i,p in reversed(list(enumerate(points))))
        parts.append(f"<polygon points='{upper} {lower}' fill='{color}' opacity='.85'/>")
    med = " ".join(f"{xy(i,p['median'])[0]:.1f},{xy(i,p['median'])[1]:.1f}" for i,p in enumerate(points))
    parts.append(f"<polyline points='{med}' fill='none' stroke='#0d5a58' stroke-width='3'/>")
    for i,p in enumerate(points):
        x,_ = xy(i,p["median"]); parts.append(f"<text x='{x:.1f}' y='{height-20}' text-anchor='middle'>{html.escape(str(p['month']))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def heat_tables(results: pd.DataFrame) -> str:
    basket = results[results["ticker"] == "Basket"]
    chunks = []
    for oni in ["Low", "Central", "High"]:
        pivot = basket[basket["oni"] == oni].pivot(index="temperature", columns="metals", values="median_return")
        pivot = pivot.reindex(index=["Low", "Central", "High"], columns=["Down", "Base", "Up"])
        rows = []
        for temp, r in pivot.iterrows():
            cells = ""
            for v in r:
                magnitude = min(abs(float(v)) / 0.15, 1.0)
                color = f"rgba(13,90,88,{0.10 + 0.45*magnitude:.3f})" if v >= 0 else f"rgba(168,79,42,{0.10 + 0.45*magnitude:.3f})"
                cells += f"<td class='heat' style='background:{color}'>{fmt_pct(v)}</td>"
            rows.append(f"<tr><th>{temp} temperature</th>{cells}</tr>")
        chunks.append(f"<section class='heat-card'><h4>ONI {oni}</h4><table><thead><tr><th></th><th>Metals down</th><th>Metals base</th><th>Metals up</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>")
    return "<div class='heat-grid'>" + "".join(chunks) + "</div>"


def make_report(results: pd.DataFrame, coeff: pd.DataFrame, fan: pd.DataFrame, regimes: pd.DataFrame,
                factors: pd.DataFrame, last_prices: dict[str, float]) -> str:
    central = results[(results["oni"] == "Central") & (results["temperature"] == "Central") & (results["metals"] == "Base")].copy()
    central["name"] = central["ticker"].map(NAMES).fillna("Equal-weight supplier basket")
    central = central.sort_values("median_return", ascending=False)
    envelope = results.groupby("ticker").agg(
        worst_scenario_median=("median_return", "min"), best_scenario_median=("median_return", "max"),
        min_prob_positive=("prob_positive", "min"), max_prob_positive=("prob_positive", "max")
    ).reset_index()
    envelope["name"] = envelope["ticker"].map(NAMES).fillna("Equal-weight supplier basket")

    forecast = json.loads(FORECAST_PATH.read_text())["forecast"]
    fc = pd.DataFrame(forecast).query("month in [10,11,12]")
    temps = weighted_state_paths()
    scenario_inputs = pd.DataFrame({
        "month": MONTHS,
        "oni_low": fc["p10_c"].to_numpy(), "oni_central": fc["oni_c"].to_numpy(), "oni_high": fc["p90_c"].to_numpy(),
        "temp_low": temps["Low"], "temp_central": temps["Central"], "temp_high": temps["High"],
    })

    model_table = coeff.copy()
    model_table["name"] = model_table["ticker"].map(NAMES)
    factor_cols = [("ticker", "Ticker", None), ("n", "Months", lambda x: f"{int(x)}"),
                   ("start", "Start", None), ("oos_r2", "Blocked-CV R²", fmt_num),
                   ("oos_corr", "OOS correlation", fmt_num), ("resid_vol", "Residual vol (ann.)", fmt_pct),
                   ("XLI", "XLI +1σ", fmt_pct), ("Copper", "Copper +1σ", fmt_pct),
                   ("Aluminum", "Al +1σ", fmt_pct), ("Nickel", "Nickel +1σ", fmt_pct),
                   ("ONI", "ONI +1σ", fmt_pct), ("dONI", "ΔONI +1σ", fmt_pct),
                   ("MidwestTemp", "Temp +1σ", fmt_pct)]

    central_table = table(central, [
        ("ticker", "Ticker", None), ("start_price", "Sep 11 close", lambda x: "$" + fmt_num(x)),
        ("median_end_price", "Dec median", lambda x: "$" + fmt_num(x)),
        ("p05_end_price", "Dec P5", lambda x: "$" + fmt_num(x)), ("p95_end_price", "Dec P95", lambda x: "$" + fmt_num(x)),
        ("median_return", "Median return", fmt_pct), ("prob_positive", "P(return > 0)", fmt_pct),
        ("prob_loss20", "P(loss > 20%)", fmt_pct), ("cvar05_return", "5% tail mean", fmt_pct),
    ])
    envelope_table = table(envelope.sort_values("worst_scenario_median", ascending=False), [
        ("ticker", "Ticker", None), ("worst_scenario_median", "Worst median", fmt_pct),
        ("best_scenario_median", "Best median", fmt_pct), ("min_prob_positive", "Lowest P(+)", fmt_pct),
        ("max_prob_positive", "Highest P(+)", fmt_pct),
    ])
    regime_basket = regimes[regimes["ticker"] == "Basket"]
    regime_table = table(regime_basket, [
        ("regime", "XLI regime, Oct–Dec", None), ("n", "Paths", lambda x: f"{int(x):,}"),
        ("median_return", "Basket median", fmt_pct), ("prob_positive", "P(basket > 0)", fmt_pct),
    ])
    input_table = table(scenario_inputs, [
        ("month", "Month", None), ("oni_low", "ONI P10", lambda x: fmt_num(x,1)),
        ("oni_central", "ONI central", lambda x: fmt_num(x,1)), ("oni_high", "ONI P90", lambda x: fmt_num(x,1)),
        ("temp_low", "Temp P10 °C", lambda x: fmt_num(x,1)), ("temp_central", "Temp central °C", lambda x: fmt_num(x,1)),
        ("temp_high", "Temp P90 °C", lambda x: fmt_num(x,1)),
    ])
    metal_frame = metal_scenario_frame(factors)
    metal_table = table(metal_frame, [
        ("scenario", "Metals case", None), ("month", "Month", None),
        ("Copper", "Copper mean", fmt_pct), ("Aluminum", "Aluminium mean", fmt_pct),
        ("Nickel", "Nickel mean", fmt_pct),
    ])
    model_html = table(model_table, factor_cols)
    detail = results.copy()
    detail["order"] = detail["ticker"].map({t: i for i, t in enumerate(TICKERS + ["Basket"])})
    detail = detail.sort_values(["oni", "temperature", "metals", "order"])
    detail_table = table(detail, [
        ("scenario", "Scenario", None), ("ticker", "Ticker", None),
        ("median_return", "Median", fmt_pct), ("p05_return", "P5", fmt_pct),
        ("p95_return", "P95", fmt_pct), ("prob_positive", "P(+)", fmt_pct),
        ("median_end_price", "Dec median price", lambda x: "$" + fmt_num(x)),
    ])

    basket_central = central[central["ticker"] == "Basket"].iloc[0]
    leader = central[central["ticker"] != "Basket"].iloc[0]
    weakest = central[central["ticker"] != "Basket"].iloc[-1]
    coeff_nonmarket = coeff.copy()
    coeff_nonmarket["climate_abs"] = coeff_nonmarket[["ONI", "dONI", "MidwestTemp"]].abs().sum(axis=1)
    climate_leader = coeff_nonmarket.sort_values("climate_abs", ascending=False).iloc[0]

    supplier = pd.read_csv(SUPPLIER_PATH)
    universe = supplier[supplier["ticker"].isin(TICKERS)].copy().sort_values("rank")
    universe_table = table(universe, [("ticker", "Ticker", None), ("company", "Company", None),
        ("role", "Cooling exposure", None), ("purity", "Exposure purity", None)])

    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Cooling suppliers: Oct–Dec 2026 scenario Monte Carlo</title>
<style>
:root{{--ink:#17282c;--muted:#607075;--paper:#f5f2ea;--card:#fffdf8;--teal:#0d5a58;--mint:#8fc7bd;--rust:#a84f2a;--line:#d7ddd9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.hero{{background:linear-gradient(135deg,#102f35,#164f50);color:white;padding:58px max(5vw,28px) 52px}} .eyebrow{{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:#b8ded8}}
h1{{font-family:Georgia,serif;font-size:clamp(38px,6vw,72px);line-height:1.02;margin:12px 0;max-width:1050px;font-weight:500}} .deck{{max-width:900px;font-size:19px;color:#d7ebe7}}
.meta{{display:flex;gap:22px;flex-wrap:wrap;margin-top:28px;font-size:13px;color:#b8d2cf}} main{{max-width:1180px;margin:auto;padding:42px 24px 80px}}
h2{{font-family:Georgia,serif;font-size:34px;font-weight:500;margin:56px 0 12px}} h3{{font-size:21px;margin:28px 0 8px}} h4{{margin:0 0 8px}} p{{max-width:900px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 28px}} .kpi,.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:0 5px 18px rgba(16,47,53,.05)}}
.kpi b{{font-size:28px;color:var(--teal);display:block}} .kpi span{{color:var(--muted);font-size:13px}} .callout{{border-left:5px solid var(--rust);background:#fff8ed;padding:17px 20px;margin:22px 0;border-radius:0 10px 10px 0;max-width:1000px}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:16px 0}} table{{border-collapse:collapse;width:100%;min-width:760px}} th,td{{padding:10px 12px;border-bottom:1px solid #e4e7e4;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} thead th{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;background:#edf4f1;color:#35575a}} tbody tr:hover{{background:#f4faf7}}
.chart{{background:white;border:1px solid var(--line);border-radius:12px;padding:12px;margin:18px 0}} svg text{{font:12px system-ui;fill:#52666a}} .heat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .heat-card{{background:white;border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto}} .heat-card table{{min-width:410px}} .heat-card td{{font-weight:700}}
.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2f0;border-radius:8px;padding:14px;overflow:auto;max-width:1000px}} .small{{font-size:13px;color:var(--muted)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} a{{color:#0b6662}} footer{{border-top:1px solid var(--line);margin-top:60px;padding-top:20px;color:var(--muted);font-size:13px}}
@media(max-width:850px){{.kpis,.two,.heat-grid{{grid-template-columns:1fr}} h1{{font-size:42px}}}}
@media print{{.hero{{background:#173f42!important;-webkit-print-color-adjust:exact}} .card,.kpi,.table-wrap{{box-shadow:none}}}}
</style></head><body>
<header class='hero'><div class='eyebrow'>Scenario research • information set frozen 11 Sep 2026</div>
<h1>Cooling equities:<br>three months, twenty-seven worlds</h1>
<p class='deck'>A 10,000-path simulation for every ONI × regional-temperature × metals scenario, applied to seven publicly traded data-center cooling suppliers and an equal-dollar basket.</p>
<div class='meta'><span>Forecast window: Oct–Dec 2026</span><span>Stock source: Massive adjusted bars</span><span>Factor history through Jul 2026</span><span>Seed: {SEED:,}</span></div></header>
<main>
<section><h2>Decision brief</h2>
<div class='kpis'><div class='kpi'><b>{fmt_pct(basket_central['median_return'])}</b><span>central-scenario basket median return</span></div>
<div class='kpi'><b>{fmt_pct(basket_central['prob_positive'])}</b><span>central P(basket finishes positive)</span></div>
<div class='kpi'><b>{leader['ticker']}</b><span>highest central median: {fmt_pct(leader['median_return'])}</span></div>
<div class='kpi'><b>{fmt_pct(basket_central['p05_return'])}</b><span>central basket 5th-percentile return</span></div></div>
<p>The central scenario does <strong>not</strong> generate a clean climate trade. {leader['ticker']} has the strongest modeled median, while {weakest['ticker']} has the weakest. The much larger spread is created by the industrial-equity regime and company residual volatility—not by the incremental ONI or temperature terms.</p>
<div class='callout'><strong>Most important interpretation.</strong> This is a conditional distribution, not a target-price model. A stock can have a positive median while still having a low signal-to-noise ratio. The climate coefficients are regularized correlations; they are not causal estimates of cooling orders, margins, or earnings.</div>
</section>

<section><h2>Central path: prices and downside</h2><p>Central means ONI central forecast, capacity-weighted Midwest temperature central effect, and the AR(1) metals baseline. XLI and unexplained company shocks are resampled from history.</p>{central_table}
<div class='chart'><h3>Equal-dollar basket fan</h3>{fan_svg(fan)}<p class='small'>Inner band: P25–P75. Outer band: P5–P95. The basket starts at 100 on the 11 September information date.</p></div></section>

<section><h2>All 27 macro-climate scenarios</h2><p>Each cell is the median Oct–Dec return of the equal-dollar supplier basket. These are stress cells, not probability-weighted forecasts.</p>{heat_tables(results)}
<h3>Company sensitivity envelope</h3><p>The table takes the lowest and highest scenario median for each company across all 27 cells. A narrow envelope means changing ONI/temperature/metals assumptions has little influence relative to simulated market and residual risk.</p>{envelope_table}
<details><summary><strong>Open all 216 company × scenario results</strong></summary>{detail_table}</details>
<p class='small'>Machine-readable version: <a href='calculated/all_27_scenario_results.csv'>all_27_scenario_results.csv</a>.</p></section>

<section><h2>The seasonal weather mechanism</h2><p>The state temperature input is the 2025 disclosed-capacity-weighted mean across Illinois, Indiana, Iowa, Michigan, North Dakota and Ohio. It is an <em>ENSO-associated deviation from a neutral counterfactual</em>, not the total anomaly from normal.</p>{input_table}
<div class='callout'><strong>Cooling implication:</strong> October is modeled cooler (central −1.2°C), November roughly neutral (−0.2°C), and the >2°C cluster appears in December (+2.8°C). Because these states remain in winter conditions and the prior engineering screen found no design-capacity breach, central weather-driven emergency equipment capex remains zero. The model therefore treats temperature primarily as a historical return covariate, not as a fabricated order uplift.</div>
<h3>Metal price paths</h3><p>July ended with copper approximately flat month over month, aluminium down 8.5% and nickel down 5.6%. The base case mean-reverts those moves through a fitted AR(1); down/up cases shift all three means by ∓0.5 historical monthly standard deviations. Empirical innovations still create dispersion around each mean.</p>{metal_table}</section>

<section><h2>What drives the distribution</h2><p>Under the central climate/metals path, conditioning on the simulated XLI regime changes the basket result far more than selecting low versus high ONI.</p>{regime_table}
<h3>Factor coefficients and model quality</h3><p>Coefficients are monthly log-return changes associated with a one historical-standard-deviation increase in each factor, after controlling for the other factors. Negative blocked-CV R² means the factor model forecasts worse than the test-fold mean; it is direct evidence that the climate/metal signal is weak.</p>{model_html}
<p class='small'>The largest regularized absolute climate loading belongs to {climate_leader['ticker']}; that is exposure, not proof. Residual annualized volatility is the portion the factor model did not explain.</p></section>

<section><h2>Method, formulas and controls</h2>
<div class='two'><div class='card'><h3>Factor model</h3><div class='formula'>rᵢ,t = μᵢ + β₁zXLI,t + β₂zCu,t + β₃zAl,t + β₄zNi,t + β₅zONI,t + β₆zΔONI,t + β₇zTemp,t + εᵢ,t</div><p>All returns are monthly log returns: <span class='formula'>rₜ = ln(Pₜ / Pₜ₋₁)</span>. Each z is a standardized predictor. Ridge chooses β by minimizing squared error plus <span class='formula'>λΣβ²</span>; λ is selected with ordered time-series folds. For the forecast, μᵢ is deliberately set to zero: historical company drift is not extrapolated.</p></div>
<div class='card'><h3>Price paths</h3><div class='formula'>Pᵢ,T = Pᵢ,0 × exp(Σₜ rᵢ,t)</div><p>Starting prices are adjusted Massive closes on 11 September 2026. Market and residual shocks use circular three-month empirical blocks. This retains observed cross-company and short-run serial dependence without assuming normal tails.</p></div></div>
<h3>Scenario construction</h3><ul>
<li><strong>ONI:</strong> low/central/high use the prior forecast's nominal P10/mean/P90 paths. Because every Oct–Dec central value exceeds the historical training maximum, regression inputs are clipped to the observed ONI support; raw values remain shown above.</li>
<li><strong>Temperature:</strong> P10/central/P90 capacity-weighted ENSO effects. The historical factor is the same states' temperature anomaly versus their 1991–2020 monthly normals.</li>
<li><strong>Metals:</strong> copper, aluminium and nickel monthly returns follow separate AR(1) baselines. Down/up shifts equal ∓0.5 historical monthly standard deviations; correlated innovations are block-resampled.</li>
<li><strong>Stock shocks:</strong> common 2020–Jul 2026 residual blocks preserve observed co-movement across the seven suppliers. Exactly {N_PATHS:,} paths are run in each of 27 cells.</li></ul>
<h3>Metric glossary</h3><ul>
<li><strong>Median return:</strong> 50th percentile of simulated terminal simple returns, <span class='formula'>median(PDec/P₀ − 1)</span>.</li>
<li><strong>P5 / P95:</strong> fifth and ninety-fifth percentiles; 90% of simulated outcomes fall between them, conditional on the scenario and model.</li>
<li><strong>P(return &gt; 0):</strong> fraction of paths with a positive terminal return. It is a model frequency, not a calibrated real-world probability.</li>
<li><strong>5% tail mean (CVaR₅):</strong> mean terminal return among paths at or below the P5 cutoff.</li>
<li><strong>Blocked-CV R²:</strong> <span class='formula'>1 − Σ(y−ŷ)² / Σ(y−ȳ)²</span> on ordered holdout folds. Values below zero indicate weak out-of-sample specification.</li>
<li><strong>OOS correlation:</strong> Pearson correlation between blocked-fold predictions and realized returns. It measures direction co-movement, not P&amp;L.</li>
<li><strong>Residual volatility:</strong> standard deviation of unexplained monthly returns × √12.</li></ul></section>

<section><h2>Universe and scope</h2>{universe_table}<p>SPXC is retained as a low-purity heat-rejection adjacency, so its result should not be read as a pure liquid-cooling exposure. Installers FIX and EME were excluded to keep the basket equipment-focused.</p></section>

<section><h2>Limits that can change the answer</h2><ol>
<li><strong>Forecast extrapolation:</strong> the Oct–Dec ONI path is above the historical seasonal maximum. Clipping prevents explosive return extrapolation but compresses differences between ONI scenarios.</li>
<li><strong>No point-in-time ONI vintage archive:</strong> the historical ONI series is revised. This is a conditional factor simulation, not a leakage-free trading backtest. A tradable historical test would require vintages or an imposed publication delay.</li>
<li><strong>Weak causal bridge:</strong> warm winter anomalies do not automatically require new cooling equipment. Procurement lead times, spare capacity, water constraints, rack density and already-placed orders matter more.</li>
<li><strong>Short corporate histories:</strong> VRT and the current TT identity begin in 2020; NVT begins in 2018. Blocked-CV results are therefore more informative than in-sample fit.</li>
<li><strong>Unmodeled valuation events:</strong> earnings, guidance, AI-capex revisions, rates, tariffs, FX, supply constraints and acquisitions can dominate a three-month stock path.</li>
<li><strong>Scenario dependence:</strong> the P10/P90 ONI and temperature paths are marginal bands, not a joint probability distribution. Do not average the 27 cells as if equally likely.</li></ol></section>

<section><h2>Data lineage</h2><ul>
<li><a href='https://massive.com/docs/rest/stocks/aggregates/custom-bars'>Massive adjusted aggregate bars</a>: monthly stock and XLI closes through 11 Sep 2026.</li>
<li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/'>NOAA CPC ONI</a>: supplied aligned series through Jul 2026; recent values may be revised.</li>
<li><a href='../enso_forecast_2026/ENSO_forecast_September_December_2026.html'>Prior calibrated ONI and state-temperature forecast</a>, information date 11 Sep 2026.</li>
<li><a href='../enso_datacenter_cooling_capex_2026/ENSO_2026_cooling_capex_supplier_research.html'>Prior cooling-capex and supplier research</a>.</li>
<li><a href='../enso_metals_walkforward/ENSO_metals_cooling_walkforward_report.html'>Prior ENSO/metals walk-forward study</a>.</li></ul></section>
<footer>Research simulation, not investment, tax or legal advice. Generated 13 Sep 2026. Prices are adjusted where provided by Massive; corporate actions and ticker-identity choices are documented in the accompanying files.</footer>
</main></body></html>"""


def main() -> None:
    CALC.mkdir(parents=True, exist_ok=True)
    factors, stock_returns, last_prices = load_factors_and_returns()
    models = {ticker: fit_ridge(factors[FEATURES], stock_returns[ticker]) for ticker in TICKERS}
    coeff = coefficient_table(models)
    results, _, fan, regimes = run_simulation(factors, models, last_prices)

    results.to_csv(CALC / "all_27_scenario_results.csv", index=False)
    coeff.to_csv(CALC / "factor_model_diagnostics.csv", index=False)
    fan.to_csv(CALC / "central_monthly_price_fan.csv", index=False)
    regimes.to_csv(CALC / "central_market_regimes.csv", index=False)
    weighted = weighted_state_paths()
    pd.DataFrame(weighted, index=MONTHS).to_csv(CALC / "temperature_scenarios.csv", index_label="month")
    oni, _ = oni_paths()
    pd.DataFrame(oni, index=MONTHS).to_csv(CALC / "oni_scenarios.csv", index_label="month")
    metal_scenario_frame(factors).to_csv(CALC / "metal_scenarios.csv", index=False)
    provenance = {
        "information_date": "2026-09-11", "generated": "2026-09-13", "seed": SEED,
        "paths_per_scenario": N_PATHS, "scenario_count": 27,
        "inputs": {},
    }
    for p in sorted(list(RAW.glob("*.json")) + [METALS_PATH, FORECAST_PATH, STATE_EFFECT_PATH, STATE_TEMP_PATH, SUPPLIER_PATH]):
        provenance["inputs"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (CALC / "provenance.json").write_text(json.dumps(provenance, indent=2))
    REPORT.write_text(make_report(results, coeff, fan, regimes, factors, last_prices))
    print(f"Wrote {REPORT}")
    print(results[(results.oni == 'Central') & (results.temperature == 'Central') & (results.metals == 'Base')][
        ['ticker','median_return','p05_return','p95_return','prob_positive']].to_string(index=False))


if __name__ == "__main__":
    main()
