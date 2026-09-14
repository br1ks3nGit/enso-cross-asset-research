#!/usr/bin/env python3
"""Backtest and three-scenario forecast for a confirmed ENSO cooling strategy."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_NEWS = ROOT / "raw/news"
CALC = ROOT / "calculated"
REPORT = ROOT / "confirmed_enso_cooling_strategy_report.html"
PRIOR = ROOT.parent
PRICE_ROOT = PRIOR / "datacenter_cooling_monte_carlo_2026/raw"
METALS_PATH = Path(os.environ.get(
    "ENSO_ONI_METALS_CSV",
    str(PRIOR / "inputs" / "enso_oni_metals" / "aligned_monthly_ONI_metals_1992_2026-07.csv"),
))
TEMP_PATH = PRIOR / "enso_forecast_2026/calculated/state_temperature.jsonl.gz"
FORECAST_PATH = PRIOR / "enso_forecast_2026/calculated/oni_forecast.json"
STATE_EFFECT_PATH = PRIOR / "enso_forecast_2026/calculated/state_enso_effects.json"
SUPPLIER_PATH = PRIOR / "enso_datacenter_cooling_capex_2026/calculated/public_supplier_screen.csv"

TICKERS = ["MOD", "AAON", "VRT", "NVT", "JCI", "TT", "SPXC"]
NAMES = {"MOD": "Modine", "AAON": "AAON", "VRT": "Vertiv", "NVT": "nVent",
         "JCI": "Johnson Controls", "TT": "Trane Technologies", "SPXC": "SPX Technologies"}
STARTS = {"MOD": "2004-11", "AAON": "2004-02", "VRT": "2020-03", "NVT": "2018-05",
          "JCI": "2016-10", "TT": "2020-04", "SPXC": "2015-10"}
CAPACITY_GW = {"Illinois": 1.746, "Indiana": 0.061, "Iowa": 0.660,
               "Michigan": 0.127, "North Dakota": 1.090, "Ohio": 0.892}
PURITY = pd.Series({"MOD": 1.00, "AAON": 1.00, "VRT": 0.85, "NVT": 0.65,
                    "JCI": 0.45, "TT": 0.50, "SPXC": 0.15})
PURITY_Z = (PURITY - PURITY.mean()) / PURITY.std(ddof=0)
FUNDAMENTAL = re.compile(
    r"(backlog|order|guidance|revenue|sales|earnings|capacity|contract|agreement|demand|data.?center|cooling|hyperscaler)", re.I
)
FORECAST_FEATURES = ["XLI", "Copper", "Aluminum", "Nickel", "enso_surprise_z", "cooling_pressure_z", "news_z", "relmom_z"]
BACKTEST_START = pd.Period("2021-06", "M")
BACKTEST_END = pd.Period("2026-08", "M")
MODEL_END = pd.Period("2026-07", "M")
N_PATHS = 20_000
SEED = 20260913
COST = 0.0015  # 15 bps per one-way unit of turnover
MONTHS = ["Oct 2026", "Nov 2026", "Dec 2026"]


def load_monthly(ticker: str) -> tuple[pd.Series, pd.Series]:
    rows = json.loads((PRICE_ROOT / f"massive_{ticker}_monthly.json").read_text())["results"]
    close = pd.Series({pd.to_datetime(r["t"], unit="ms").to_period("M"): float(r["c"]) for r in rows}).sort_index()
    volume = pd.Series({pd.to_datetime(r["t"], unit="ms").to_period("M"): float(r.get("v", np.nan)) for r in rows}).sort_index()
    return close, volume


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    prices, volumes = {}, {}
    for ticker in TICKERS + ["XLI"]:
        prices[ticker], volumes[ticker] = load_monthly(ticker)
    prices = pd.DataFrame(prices).sort_index()
    volumes = pd.DataFrame(volumes).sort_index()
    log_returns = np.log(prices).diff()
    simple_returns = prices.pct_change(fill_method=None)
    last_prices = {ticker: float(prices[ticker].dropna().iloc[-1]) for ticker in TICKERS}
    return prices, log_returns, simple_returns, last_prices


def load_oni_metals() -> pd.DataFrame:
    raw = pd.read_csv(METALS_PATH)
    raw.index = pd.to_datetime(raw["Date"]).dt.to_period("M")
    out = pd.DataFrame(index=raw.index)
    out["ONI"] = raw["ONI"].astype(float)
    out["dONI"] = out["ONI"].diff()
    for metal in ["Copper", "Aluminum", "Nickel"]:
        out[metal] = np.log(raw[metal].astype(float)).diff().to_numpy()
    return out


def oni_surprises(oni: pd.Series) -> pd.DataFrame:
    rows, past_errors = [], []
    for i in range(61, len(oni)):
        history = oni.iloc[max(0, i - 120):i].dropna()
        x, y = history.iloc[:-1].to_numpy(), history.iloc[1:].to_numpy()
        phi, intercept = np.polyfit(x, y, 1)
        prediction = float(intercept + phi * oni.iloc[i - 1])
        error = float(oni.iloc[i] - prediction)
        recent = np.array(past_errors[-60:])
        z = float((error - recent.mean()) / recent.std(ddof=1)) if len(recent) >= 24 and recent.std(ddof=1) > 0 else np.nan
        source = oni.index[i]
        rows.append({"source_month": source, "trade_month": source + 3, "latest_oni": float(oni.iloc[i]),
                     "latest_doni": float(oni.iloc[i] - oni.iloc[i - 1]), "expected_oni": prediction,
                     "surprise": error, "enso_surprise_z": z})
        past_errors.append(error)
    return pd.DataFrame(rows).set_index("trade_month")


def capacity_cdd() -> pd.Series:
    data = pd.read_json(TEMP_PATH, lines=True)
    data = data[data["state"].isin(CAPACITY_GW)].copy()
    data["weight"] = data["state"].map(CAPACITY_GW)
    data["cdd_proxy"] = np.maximum(data["temperature_c"] - 18.0, 0.0)
    grouped = data.groupby(["year", "month"]).apply(
        lambda g: np.average(g["cdd_proxy"], weights=g["weight"]), include_groups=False
    )
    grouped.index = pd.PeriodIndex([f"{y:04d}-{m:02d}" for y, m in grouped.index], freq="M")
    return grouped.sort_index().rename("cdd_proxy")


def cooling_forecast_series(oni: pd.Series) -> pd.DataFrame:
    cdd = capacity_cdd()
    normal_data = cdd[(cdd.index.year >= 1991) & (cdd.index.year <= 2020)]
    normals = normal_data.groupby(normal_data.index.month).mean()
    excess = pd.Series({p: float(v - normals.loc[p.month]) for p, v in cdd.items()}, name="cdd_excess")
    rows, prior_predictions = [], []
    for target in pd.period_range("1995-01", "2026-12", freq="M"):
        source = target - 3
        pairs = []
        for p, y in excess.items():
            if p >= target or p.month != target.month or p.year < 1992 or (p - 3) not in oni.index:
                continue
            pairs.append((float(oni.loc[p - 3]), float(y)))
        if len(pairs) < 20 or source not in oni.index:
            continue
        arr = np.array(pairs)
        x, y = arr[:, 0], arr[:, 1]
        # Small ridge penalty stabilizes month-specific physical sensitivity.
        xbar, ybar = x.mean(), y.mean()
        beta = float(np.sum((x - xbar) * (y - ybar)) / (np.sum((x - xbar) ** 2) + 5.0))
        prediction = float(ybar + beta * (oni.loc[source] - xbar))
        past = np.array(prior_predictions[-120:])
        z = float((prediction - past.mean()) / past.std(ddof=1)) if len(past) >= 24 and past.std(ddof=1) > 0 else 0.0
        rows.append({"trade_month": target, "cooling_pressure": max(prediction, 0.0), "cooling_pressure_z": z,
                     "raw_cdd_forecast": prediction, "cdd_beta": beta, "training_years": len(pairs)})
        prior_predictions.append(prediction)
    return pd.DataFrame(rows).set_index("trade_month")


def news_scores() -> pd.DataFrame:
    series = {}
    for ticker in TICKERS:
        rows = json.loads((RAW_NEWS / f"massive_news_{ticker}_2020_2026.json").read_text())["results"]
        observations = []
        for article in rows:
            text = " ".join([article.get("title", ""), article.get("description") or "", " ".join(article.get("keywords") or [])])
            if not FUNDAMENTAL.search(text):
                continue
            score = 0
            for insight in article.get("insights") or []:
                if insight.get("ticker") == ticker:
                    score = {"positive": 1, "negative": -1, "neutral": 0}.get(insight.get("sentiment"), 0)
                    break
            observations.append((pd.Timestamp(article["published_utc"]).tz_convert(None).to_period("M"), score))
        if observations:
            frame = pd.DataFrame(observations, columns=["month", "score"])
            monthly = frame.groupby("month").agg(score=("score", "sum"), articles=("score", "size"))
            series[ticker] = monthly["score"] / np.sqrt(monthly["articles"])
        else:
            series[ticker] = pd.Series(dtype=float)
    raw = pd.DataFrame(series).fillna(0.0).sort_index()
    # At the start of month t, only news through month t-1 enters the signal.
    available = raw.shift(1)
    mean = available.mean(axis=1)
    sd = available.std(axis=1, ddof=0).replace(0, np.nan)
    z = available.sub(mean, axis=0).div(sd, axis=0).fillna(0.0)
    return z


def build_signal_panel(prices: pd.DataFrame, returns: pd.DataFrame, oni_metals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enso = oni_surprises(oni_metals["ONI"])
    cooling = cooling_forecast_series(oni_metals["ONI"])
    news = news_scores()
    months = pd.period_range(BACKTEST_START, pd.Period("2026-10", "M"), freq="M")
    base = pd.DataFrame(index=months).join(enso).join(cooling)
    xli_prev = prices["XLI"].shift(1)
    base["market_on"] = (xli_prev > xli_prev.rolling(6).mean()) & (np.log(xli_prev / xli_prev.shift(3)) > 0)
    base["climate_on"] = (base["latest_oni"] > 0.5) & ((base["latest_doni"] > 0) | (base["enso_surprise_z"] > 0))

    relmom = pd.DataFrame(index=prices.index, columns=TICKERS, dtype=float)
    for ticker in TICKERS:
        relmom[ticker] = np.log(prices[ticker].shift(1) / prices[ticker].shift(4)) - np.log(prices["XLI"].shift(1) / prices["XLI"].shift(4))
    rel_mean, rel_sd = relmom.mean(axis=1), relmom.std(axis=1, ddof=0).replace(0, np.nan)
    rel_z = relmom.sub(rel_mean, axis=0).div(rel_sd, axis=0).fillna(0.0)
    news = news.reindex(months).fillna(0.0)
    rel_z = rel_z.reindex(months).fillna(0.0)
    return base, news, rel_z


def rolling_beta(stock: pd.Series, market: pd.Series, month: pd.Period) -> float:
    pair = pd.concat([stock.rename("s"), market.rename("m")], axis=1).loc[:month - 1].dropna().tail(24)
    if len(pair) < 12 or pair["m"].var() == 0:
        return 1.0
    return float(np.clip(pair["s"].cov(pair["m"]) / pair["m"].var(), 0.25, 2.0))


def backtest(prices: pd.DataFrame, returns: pd.DataFrame, signals: pd.DataFrame, news: pd.DataFrame, relmom: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, positions = [], []
    prev = {ticker: 0.0 for ticker in TICKERS}
    prev_hedge = 0.0
    naive_prev = {ticker: 0.0 for ticker in TICKERS}
    for month in pd.period_range(BACKTEST_START, BACKTEST_END, freq="M"):
        sig = signals.loc[month]
        score = 0.35 * news.loc[month] + 0.45 * relmom.loc[month] + 0.20 * PURITY_Z
        confirmation = ((news.loc[month] > 0) | (relmom.loc[month] > 0)) & PURITY.ge(0.40)
        available = [t for t in TICKERS if month in returns.index and pd.notna(returns.loc[month, t]) and confirmation[t]]
        selected = sorted(available, key=lambda t: score[t], reverse=True)[:2] if bool(sig["climate_on"]) else []
        weights = {ticker: 0.0 for ticker in TICKERS}
        gross = 0.0
        if selected:
            gross = 1.0 if bool(sig["market_on"]) else 0.5
            if not (pd.notna(sig["cooling_pressure"]) and sig["cooling_pressure"] > 0):
                gross *= 0.75
            vols = {}
            for ticker in selected:
                hist = returns[ticker].loc[:month - 1].dropna().tail(6)
                vols[ticker] = float(hist.std(ddof=1)) if len(hist) >= 4 and hist.std(ddof=1) > 0 else 0.10
            inv = {t: 1 / vols[t] for t in selected}
            total = sum(inv.values())
            for ticker in selected:
                weights[ticker] = gross * inv[ticker] / total
        betas = {t: rolling_beta(returns[t], returns["XLI"], month) for t in selected}
        full_beta_hedge = sum(weights[t] * betas[t] for t in selected)
        hedge = (0.5 if bool(sig["market_on"]) else 1.0) * full_beta_hedge
        turnover = sum(abs(weights[t] - prev[t]) for t in TICKERS) + abs(hedge - prev_hedge)
        gross_return = sum(weights[t] * float(returns.loc[month, t]) for t in TICKERS) - hedge * float(returns.loc[month, "XLI"])
        strategy_return = gross_return - COST * turnover

        naive_active = bool(sig["latest_oni"] > 0.5)
        naive_names = [t for t in TICKERS if pd.notna(returns.loc[month, t])]
        naive = {t: (1 / len(naive_names) if naive_active and t in naive_names else 0.0) for t in TICKERS}
        naive_turn = sum(abs(naive[t] - naive_prev[t]) for t in TICKERS)
        naive_return = sum(naive[t] * float(returns.loc[month, t]) for t in TICKERS) - COST * naive_turn
        basket_return = float(returns.loc[month, TICKERS].dropna().mean())
        rows.append({"month": str(month), "strategy": strategy_return, "strategy_gross": gross_return,
                     "naive_oni_basket": naive_return, "cooling_basket": basket_return, "XLI": float(returns.loc[month, "XLI"]),
                     "active": bool(selected), "selected": ", ".join(selected) if selected else "Cash",
                     "gross_exposure": gross, "xli_hedge": hedge, "turnover": turnover,
                     "latest_oni": sig["latest_oni"], "enso_surprise_z": sig["enso_surprise_z"],
                     "cooling_pressure": sig["cooling_pressure"], "market_on": bool(sig["market_on"])})
        for ticker in TICKERS:
            positions.append({"month": str(month), "ticker": ticker, "weight": weights[ticker],
                              "news_z": news.loc[month, ticker], "relmom_z": relmom.loc[month, ticker],
                              "selection_score": score[ticker], "selected": ticker in selected})
        prev, prev_hedge, naive_prev = weights, hedge, naive
    return pd.DataFrame(rows), pd.DataFrame(positions)


def hac_mean_test(series: pd.Series, lags=3) -> tuple[float, float]:
    x = series.dropna().to_numpy(float)
    e = x - x.mean(); n = len(x)
    lv = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(e[lag:] @ e[:-lag]) / n
        lv += 2 * (1 - lag / (lags + 1)) * gamma
    se = math.sqrt(max(lv, 0) / n)
    if se == 0:
        return np.nan, np.nan
    t = float(x.mean() / se)
    return t, float(2 * stats.t.sf(abs(t), df=n - 1))


def performance(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in [("strategy", "Confirmed, regime-adjusted hedge"), ("naive_oni_basket", "Naive ONI long"),
                          ("cooling_basket", "Cooling basket"), ("XLI", "XLI")]:
        r = backtest[column].astype(float)
        if column == "strategy":
            active_mask = backtest["active"].astype(bool)
        elif column == "naive_oni_basket":
            active_mask = backtest["latest_oni"].fillna(-99).gt(0.5)
        else:
            active_mask = pd.Series(True, index=backtest.index)
        wealth = (1 + r).cumprod(); n = len(r)
        dd = wealth / wealth.cummax() - 1
        t, p = hac_mean_test(r)
        rows.append({"strategy": label, "months": n, "invested_months": int(active_mask.sum()),
                     "total_return": float(wealth.iloc[-1] - 1),
                     "annual_return": float(wealth.iloc[-1] ** (12 / n) - 1),
                     "annual_vol": float(r.std(ddof=1) * np.sqrt(12)),
                     "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)) if r.std(ddof=1) else np.nan,
                     "max_drawdown": float(dd.min()),
                     "active_hit_rate": float((r[active_mask] > 0).mean()) if active_mask.any() else np.nan,
                     "hac_t": t, "hac_p": p})
    return pd.DataFrame(rows)


def fit_forecast_models(log_returns: pd.DataFrame, oni_metals: pd.DataFrame, signals: pd.DataFrame,
                        news: pd.DataFrame, relmom: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    factors = oni_metals[["Copper", "Aluminum", "Nickel"]].copy()
    factors["XLI"] = log_returns["XLI"]
    factors = factors.join(signals[["enso_surprise_z", "cooling_pressure_z"]]).loc[:MODEL_END]
    models, diagnostics = {}, []
    for ticker in TICKERS:
        x = factors.join(news[ticker].rename("news_z")).join(relmom[ticker].rename("relmom_z"))
        target = (log_returns[ticker] - log_returns["XLI"]).rename("target")
        frame = x.join(target).loc[max(pd.Period(STARTS[ticker], "M"), pd.Period("2021-06", "M")):MODEL_END].dropna()
        X, y = frame[FORECAST_FEATURES].to_numpy(float), frame["target"].to_numpy(float)
        scaler = StandardScaler().fit(X)
        # Fixed moderate shrinkage avoids selecting a penalty on fewer than 62 observations.
        model = Ridge(alpha=20.0, fit_intercept=False).fit(scaler.transform(X), y - y.mean())
        fitted = y.mean() + model.predict(scaler.transform(X))
        splitter = TimeSeriesSplit(n_splits=4)
        truth, prediction = [], []
        for train, test in splitter.split(X):
            sc = StandardScaler().fit(X[train]); mean = y[train].mean()
            mdl = Ridge(alpha=20.0, fit_intercept=False).fit(sc.transform(X[train]), y[train] - mean)
            truth.extend(y[test]); prediction.extend(mean + mdl.predict(sc.transform(X[test])))
        residual = pd.Series(y - fitted, index=frame.index, name=ticker)
        models[ticker] = {"scaler": scaler, "model": model, "residual": residual}
        diagnostics.append({"ticker": ticker, "n": len(frame), "start": str(frame.index[0]),
                            "in_r2": r2_score(y, fitted), "oos_r2": r2_score(truth, prediction),
                            "oos_corr": float(np.corrcoef(truth, prediction)[0, 1]),
                            "resid_vol": float(residual.std(ddof=1) * np.sqrt(12)),
                            **dict(zip(FORECAST_FEATURES, model.coef_))})
    return models, pd.DataFrame(diagnostics)


def ar1_metal_paths(oni_metals: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    means, residuals = [], []
    for col in ["Copper", "Aluminum", "Nickel"]:
        r = oni_metals[col].dropna()
        x, y = r.iloc[:-1].to_numpy(), r.iloc[1:].to_numpy()
        phi, intercept = np.polyfit(x, y, 1); phi = float(np.clip(phi, -.5, .5)); intercept = float(y.mean() - phi*x.mean())
        current, path = float(r.iloc[-1]), []
        for _ in range(3):
            current = intercept + phi*current; path.append(current)
        means.append(path)
        residuals.append(pd.Series(y - (intercept + phi*x), index=r.index[1:], name=col))
    return np.array(means).T, pd.concat(residuals, axis=1).dropna()


def sample_blocks(values: np.ndarray, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    starts = rng.integers(0, len(values), size=n)
    idx = (starts[:, None] + np.arange(length)[None, :]) % len(values)
    return values[idx]


def current_cross_section(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    current = prices.index.max()
    rel = pd.Series({t: np.log(prices.loc[current, t] / prices[t].dropna().iloc[-4]) -
                        np.log(prices.loc[current, "XLI"] / prices["XLI"].dropna().iloc[-4]) for t in TICKERS})
    rel_z = (rel - rel.mean()) / rel.std(ddof=0)
    # September news is partial but available at the information date.
    raw = {}
    for ticker in TICKERS:
        articles = json.loads((RAW_NEWS / f"massive_news_{ticker}_2020_2026.json").read_text())["results"]
        score = 0
        for article in articles:
            stamp = pd.Timestamp(article["published_utc"]).tz_convert(None)
            if stamp < pd.Timestamp("2026-08-13") or stamp > pd.Timestamp("2026-09-12 23:59:59"):
                continue
            text = " ".join([article.get("title", ""), article.get("description") or ""])
            if not FUNDAMENTAL.search(text): continue
            for insight in article.get("insights") or []:
                if insight.get("ticker") == ticker:
                    score += {"positive": 1, "negative": -1, "neutral": 0}.get(insight.get("sentiment"), 0)
        raw[ticker] = score
    news = pd.Series(raw, dtype=float)
    news_z = (news - news.mean()) / news.std(ddof=0) if news.std(ddof=0) else news*0
    return news_z.fillna(0), rel_z.fillna(0)


def forecast_scenarios(prices: pd.DataFrame, log_returns: pd.DataFrame, oni_metals: pd.DataFrame,
                       signals: pd.DataFrame, news_history: pd.DataFrame, models: dict,
                       last_prices: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    market_hist = log_returns["XLI"].loc["2004-02":MODEL_END].dropna().to_numpy(float)
    market = sample_blocks(market_hist, N_PATHS, 3, rng)[:, :, 0]
    metal_mean, metal_resid = ar1_metal_paths(oni_metals)
    metals = metal_mean[None, :, :] + sample_blocks(metal_resid.to_numpy(float), N_PATHS, 3, rng)
    common_resid = pd.concat({t: models[t]["residual"] for t in TICKERS}, axis=1).dropna()
    common_resid = common_resid - common_resid.mean()
    residuals = sample_blocks(common_resid.to_numpy(float), N_PATHS, 3, rng)
    current_news, current_mom = current_cross_section(prices)
    current_surprise = float(signals.loc[pd.Period("2026-10", "M"), "enso_surprise_z"])
    p90_surprise = float(signals["enso_surprise_z"].loc[:pd.Period("2026-10", "M")].quantile(.90))
    p95_cooling = float(signals["cooling_pressure_z"].loc[:pd.Period("2026-10", "M")].quantile(.95))
    purity_bonus = pd.Series({"MOD": 1.0, "AAON": 1.0, "VRT": 1.0, "NVT": .6, "JCI": .3, "TT": .4, "SPXC": 0.0})
    excellent_drift = {}
    for ticker in TICKERS:
        sample = pd.concat([(log_returns[ticker] - log_returns["XLI"]).rename("excess"),
                            news_history[ticker].rename("news")], axis=1).dropna()
        positive = sample[(sample["news"] > 0) & (sample["excess"] > 0)]["excess"]
        calibrated = float(np.clip(positive.median() if len(positive) else .01, .005, .03))
        excellent_drift[ticker] = calibrated * (0.5 + 0.5*PURITY[ticker])
    scenario_defs = {
        "Excellent": {
            "description": "Large forecast revisions are followed by confirmed cooling orders and spending.",
            "enso": np.array([max(current_surprise, p90_surprise), p90_surprise, p90_surprise]),
            "cool": np.array([p95_cooling, p95_cooling, p95_cooling]),
            "news": np.vstack([(current_news + purity_bonus).clip(-3, 3),
                               (0.5*current_news + purity_bonus).clip(-3, 3), purity_bonus]).T,
            "active": True, "gross": 1.0, "repricing": pd.Series(excellent_drift),
        },
        "Normal": {
            "description": "El Niño persists, but no emergency winter cooling-capex response is assumed.",
            "enso": np.array([current_surprise, 0.5*current_surprise, 0.25*current_surprise]),
            "cool": np.zeros(3),
            "news": np.vstack([current_news, 0.5*current_news, np.zeros(len(TICKERS))]).T,
            "active": True, "gross": .75, "repricing": pd.Series(0.0, index=TICKERS),
        },
        "Conservative": {
            "description": "No incremental ENSO earnings effect; prices follow market, metals and current company trends.",
            "enso": np.zeros(3), "cool": np.zeros(3),
            "news": np.vstack([current_news, 0.25*current_news, np.zeros(len(TICKERS))]).T,
            "active": False, "gross": 0.0, "repricing": pd.Series(0.0, index=TICKERS),
        },
    }
    rows, portfolio_rows, assumption_rows = [], [], []
    for scenario, spec in scenario_defs.items():
        stock_paths = {}
        deterministic_scores = []
        for k in range(3):
            momentum = current_mom * (0.5 ** k)
            news_k = pd.Series(spec["news"][:, k], index=TICKERS)
            score = .35*news_k + .45*momentum + .20*PURITY_Z
            deterministic_scores.append(score)
            assumption_rows.append({"scenario": scenario, "month": MONTHS[k], "enso_surprise_z": spec["enso"][k],
                                    "cooling_pressure_z": spec["cool"][k], "gross_target": spec["gross"] if spec["active"] else 0,
                                    "mean_monthly_repricing_drift": float(spec["repricing"].mean()),
                                    "description": spec["description"]})
        for j, ticker in enumerate(TICKERS):
            X = np.empty((N_PATHS, 3, len(FORECAST_FEATURES)))
            X[:, :, 0] = market; X[:, :, 1:4] = metals
            X[:, :, 4] = spec["enso"][None, :]; X[:, :, 5] = spec["cool"][None, :]
            X[:, :, 6] = spec["news"][TICKERS.index(ticker), :][None, :]
            X[:, :, 7] = np.array([current_mom[ticker]*(0.5**k) for k in range(3)])[None, :]
            m = models[ticker]
            excess = m["model"].predict(m["scaler"].transform(X.reshape(-1, len(FORECAST_FEATURES)))).reshape(N_PATHS, 3)
            stock_log = market + excess + residuals[:, :, j] + float(spec["repricing"][ticker])
            path = last_prices[ticker] * np.exp(np.cumsum(stock_log, axis=1))
            stock_paths[ticker] = path
            end_ret = path[:, -1] / last_prices[ticker] - 1
            rows.append({"scenario": scenario, "ticker": ticker, "start_price": last_prices[ticker],
                         "median_end_price": np.median(path[:, -1]), "p10_end_price": np.quantile(path[:, -1], .10),
                         "p90_end_price": np.quantile(path[:, -1], .90), "median_return": np.median(end_ret),
                         "p10_return": np.quantile(end_ret, .10), "p90_return": np.quantile(end_ret, .90),
                         "prob_positive": (end_ret > 0).mean(),
                         "monthly_repricing_drift": float(spec["repricing"][ticker])})

        # Deterministic cross-sectional selection, then pathwise beta hedge.
        portfolio = np.zeros((N_PATHS, 3)); holdings = []
        prev_weights = {t: 0.0 for t in TICKERS}; prev_hedge = 0.0
        for k in range(3):
            if spec["active"]:
                eligible = [t for t in TICKERS if PURITY[t] >= .40 and
                            (spec["news"][TICKERS.index(t), k] > 0 or current_mom[t]*(0.5**k) > 0)]
                selected = sorted(eligible, key=lambda t: deterministic_scores[k][t], reverse=True)[:2]
            else:
                selected = []
            weights = {t: 0.0 for t in TICKERS}
            if selected:
                inv = {t: 1 / max(float(log_returns[t].loc[:MODEL_END].tail(6).std(ddof=1)), .02) for t in selected}
                for t in selected: weights[t] = spec["gross"] * inv[t] / sum(inv.values())
            betas = {t: rolling_beta(log_returns[t], log_returns["XLI"], pd.Period("2026-09", "M")) for t in selected}
            hedge = 0.5 * sum(weights[t]*betas[t] for t in selected)
            stock_simple = {t: stock_paths[t][:, k] / (last_prices[t] if k == 0 else stock_paths[t][:, k-1]) - 1 for t in selected}
            market_simple = np.exp(market[:, k]) - 1
            gross_ret = sum(weights[t]*stock_simple[t] for t in selected) - hedge*market_simple
            turnover = sum(abs(weights[t]-prev_weights[t]) for t in TICKERS) + abs(hedge-prev_hedge)
            portfolio[:, k] = gross_ret - COST*turnover
            holdings.append(", ".join(selected) if selected else "Cash")
            prev_weights, prev_hedge = weights, hedge
        cumulative = np.prod(1 + portfolio, axis=1) - 1
        portfolio_rows.append({"scenario": scenario, "holdings_oct": holdings[0], "holdings_nov": holdings[1], "holdings_dec": holdings[2],
                               "median_return": np.median(cumulative), "p10_return": np.quantile(cumulative, .10),
                               "p90_return": np.quantile(cumulative, .90), "prob_positive": (cumulative > 0).mean(),
                               "prob_loss10": (cumulative < -.10).mean()})
    return pd.DataFrame(rows), pd.DataFrame(portfolio_rows), pd.DataFrame(assumption_rows)


def fmt_pct(x, digits=1): return "n.a." if pd.isna(x) else f"{100*float(x):.{digits}f}%"
def fmt_num(x, digits=2): return "n.a." if pd.isna(x) else f"{float(x):,.{digits}f}"


def table(frame: pd.DataFrame, columns: list[tuple[str, str, object]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, formatter in columns:
            value = row.get(key); rendered = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def wealth_svg(backtest: pd.DataFrame, width=920, height=390) -> str:
    labels = {"strategy": "Confirmed hedge", "naive_oni_basket": "Naive ONI", "cooling_basket": "Cooling basket", "XLI": "XLI"}
    colors = {"strategy": "#0b5d59", "naive_oni_basket": "#bb6136", "cooling_basket": "#69777c", "XLI": "#233f68"}
    wealth = pd.DataFrame({c: (1+backtest[c]).cumprod()*100 for c in labels})
    lo, hi = float(wealth.min().min()), float(wealth.max().max()); pad=(hi-lo)*.1;lo-=pad;hi+=pad
    left,right,top,bottom=58,24,34,50
    def xy(i,v): return (left+i*(width-left-right)/(len(wealth)-1), top+(hi-v)*(height-top-bottom)/(hi-lo))
    out=[f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Backtest wealth comparison'>"]
    for frac in np.linspace(0,1,5):
        v=lo+frac*(hi-lo);y=xy(0,v)[1];out.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#dde1de'/><text x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{v:.0f}</text>")
    for c,label in labels.items():
        pts=" ".join(f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}" for i,v in enumerate(wealth[c]))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{colors[c]}' stroke-width='2.5'/>")
    x=left
    for c,label in labels.items(): out.append(f"<line x1='{x}' x2='{x+22}' y1='17' y2='17' stroke='{colors[c]}' stroke-width='3'/><text x='{x+27}' y='21'>{label}</text>");x+=165
    ticks=[0,len(wealth)//2,len(wealth)-1]
    for i in ticks: out.append(f"<text x='{xy(i,lo)[0]:.1f}' y='{height-18}' text-anchor='middle'>{backtest.month.iloc[i]}</text>")
    out.append("</svg>");return "".join(out)


def report_html(backtest: pd.DataFrame, positions: pd.DataFrame, perf: pd.DataFrame, forecast: pd.DataFrame,
                portfolio: pd.DataFrame, assumptions: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    active = backtest[backtest["active"]].copy()
    active_table = table(active, [("month","Test month",None),("latest_oni","Available ONI",lambda x:fmt_num(x,1)),
        ("enso_surprise_z","ENSO surprise z",fmt_num),("cooling_pressure","CDD increment",fmt_num),
        ("market_on","XLI trend positive",lambda x:"Yes" if x else "No"),("selected","Selected",None),
        ("strategy","Net return",fmt_pct),("xli_hedge","XLI hedge",fmt_num)])
    perf_table = table(perf, [("strategy","Portfolio",None),("total_return","Total return",fmt_pct),
        ("annual_return","Annualized return",fmt_pct),("annual_vol","Annualized vol",fmt_pct),
        ("sharpe","Sharpe",fmt_num),("max_drawdown","Maximum drawdown",fmt_pct),
        ("invested_months","Invested months",lambda x:str(int(x))),("active_hit_rate","Active hit rate",fmt_pct),
        ("hac_p","HAC mean p-value",fmt_num)])
    portfolio_table = table(portfolio, [("scenario","Scenario",None),("holdings_oct","October",None),
        ("holdings_nov","November",None),("holdings_dec","December",None),("median_return","Median return",fmt_pct),
        ("p10_return","P10",fmt_pct),("p90_return","P90",fmt_pct),("prob_positive","P(return > 0)",fmt_pct),
        ("prob_loss10","P(loss > 10%)",fmt_pct)])
    forecast_table = table(forecast, [("scenario","Scenario",None),("ticker","Ticker",None),
        ("start_price","11 Sep price",lambda x:"$"+fmt_num(x)),("median_end_price","Dec median",lambda x:"$"+fmt_num(x)),
        ("p10_end_price","Dec P10",lambda x:"$"+fmt_num(x)),("p90_end_price","Dec P90",lambda x:"$"+fmt_num(x)),
        ("monthly_repricing_drift","Scenario drift/month",fmt_pct),
        ("median_return","Median return",fmt_pct),("prob_positive","P(return > 0)",fmt_pct)])
    assumption_table = table(assumptions, [("scenario","Scenario",None),("month","Month",None),
        ("enso_surprise_z","ENSO surprise z",fmt_num),("cooling_pressure_z","Cooling confirmation z",fmt_num),
        ("mean_monthly_repricing_drift","Mean repricing/month",fmt_pct),
        ("gross_target","Strategy gross",fmt_pct),("description","Definition",None)])
    diag_table = table(diagnostics, [("ticker","Ticker",None),("n","Months",lambda x:str(int(x))),
        ("oos_r2","Blocked-CV R²",fmt_num),("oos_corr","OOS correlation",fmt_num),
        ("resid_vol","Residual vol",fmt_pct),("enso_surprise_z","ENSO +1σ",fmt_pct),
        ("cooling_pressure_z","Cooling +1σ",fmt_pct),("news_z","News +1σ",fmt_pct),("relmom_z","Momentum +1σ",fmt_pct)])
    strategy_perf=perf.iloc[0]; naive_perf=perf.iloc[1]; basket_perf=perf.iloc[2]; xli_perf=perf.iloc[3]
    best_scenario=portfolio.sort_values('median_return',ascending=False).iloc[0]
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Confirmed ENSO cooling strategy: backtest and 2026 scenarios</title><style>
:root{{--ink:#1c292c;--muted:#627174;--paper:#f4f1e9;--card:#fffefa;--navy:#173b52;--teal:#0b5d59;--rust:#bb6136;--line:#d8dedb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.56 Inter,system-ui,-apple-system,sans-serif}}header{{padding:56px max(5vw,26px);background:linear-gradient(135deg,#122f3d,#145750);color:white}}h1{{font:500 clamp(38px,6vw,68px)/1.03 Georgia,serif;margin:10px 0;max-width:1050px}}.deck{{font-size:19px;color:#d5e7e3;max-width:900px}}.meta{{margin-top:25px;color:#b9d4d0;display:flex;gap:22px;flex-wrap:wrap;font-size:13px}}main{{max-width:1180px;margin:auto;padding:42px 24px 80px}}h2{{font:500 34px Georgia,serif;margin:54px 0 12px}}h3{{font-size:21px;margin:28px 0 8px}}p{{max-width:920px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.kpi,.card{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:19px;box-shadow:0 5px 18px rgba(20,55,55,.05)}}.kpi b{{display:block;font-size:28px;color:var(--teal)}}.kpi span,.small{{font-size:13px;color:var(--muted)}}.callout{{border-left:5px solid var(--rust);background:#fff8ed;padding:17px 20px;border-radius:0 9px 9px 0;margin:20px 0;max-width:1000px}}.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:9px;margin:14px 0}}table{{border-collapse:collapse;width:100%;min-width:780px}}th,td{{padding:10px 12px;border-bottom:1px solid #e3e7e4;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#eaf2ef;color:#35565a;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}.chart{{background:white;border:1px solid var(--line);border-radius:11px;padding:12px}}svg text{{font:12px system-ui;fill:#52666a}}.formula{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2ef;padding:12px;border-radius:7px;overflow:auto}}.three{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}a{{color:#0a6661}}li{{margin:7px 0}}footer{{margin-top:55px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}@media(max-width:850px){{.kpis,.three{{grid-template-columns:1fr}}}}
</style></head><body><header><div class='small' style='color:#b9d4d0;text-transform:uppercase;letter-spacing:.13em'>Information-frozen signal design</div><h1>Confirmed ENSO cooling strategy</h1><p class='deck'>A monthly backtest that trades forecast surprise only when company news or relative momentum confirms it, scales exposure by cooling-load and market conditions, and hedges XLI beta.</p><div class='meta'><span>Backtest: Jun 2021–Aug 2026</span><span>Forecast: Oct–Dec 2026</span><span>Massive adjusted prices and news</span><span>15 bp one-way transaction cost</span></div></header><main>
<section><h2>Result</h2><div class='kpis'><div class='kpi'><b>{fmt_pct(strategy_perf.total_return)}</b><span>confirmed strategy total return</span></div><div class='kpi'><b>{fmt_num(strategy_perf.sharpe)}</b><span>monthly Sharpe, annualized</span></div><div class='kpi'><b>{int(active.shape[0])}</b><span>active test months</span></div><div class='kpi'><b>{fmt_pct(best_scenario.median_return)}</b><span>{best_scenario.scenario} forecast median</span></div></div>
<p>The strategy converts the earlier conclusions into explicit rules. It does not buy a high ONI level by itself. It requires a positive information surprise, company confirmation and relative strength, reduces exposure when XLI weakens, and uses a regime-adjusted beta hedge.</p><div class='callout'><strong>Backtest conclusion.</strong> The new rules improved capital preservation, not total profit. Maximum drawdown was {fmt_pct(strategy_perf.max_drawdown)} versus {fmt_pct(naive_perf.max_drawdown)} for the naive ONI basket, but total return was {fmt_pct(strategy_perf.total_return)} versus {fmt_pct(naive_perf.total_return)}. The mean-return HAC p-value was {fmt_num(strategy_perf.hac_p)}, so the strategy return is not statistically distinguishable from zero at conventional thresholds.</div><div class='callout'><strong>Evidence limit.</strong> The full news-confirmation record starts in 2021, leaving only one completed strong El Niño and the developing 2026 event. The backtest therefore has {int(active.shape[0])} active months. Treat the result as a specification check, not a reliable expected-return estimate. The cooling basket's {fmt_pct(basket_perf.total_return)} return mainly reflects the AI-infrastructure equity rally and survivorship, not demonstrated ENSO alpha.</div>{perf_table}<div class='chart'>{wealth_svg(backtest)}</div></section>
<section><h2>Active walk-forward decisions</h2><p>Every row uses information available before the test month. ONI is delayed by three months, news and price confirmation are lagged one month, and betas and volatility use only earlier returns.</p>{active_table}</section>
<section><h2>Forecast scenarios: October–December 2026</h2><div class='three'><div class='card'><h3>Excellent</h3><p>Very strong ENSO revisions translate into confirmed cooling orders. Because this case explicitly assumes prices rise, it adds each ticker's historical median positive news-response drift, capped at 3% monthly and scaled by cooling purity.</p></div><div class='card'><h3>Normal</h3><p>El Niño persists, but the Midwest winter anomaly does not create emergency cooling-equipment capex. Current news and momentum fade.</p></div><div class='card'><h3>Conservative</h3><p>No incremental ENSO earnings effect. Company prices retain market, metals and current company-specific variability; the climate strategy stays in cash.</p></div></div><h3>Strategy forecast</h3>{portfolio_table}<h3>Company price distributions</h3>{forecast_table}<h3>Scenario inputs</h3>{assumption_table}</section>
<section><h2>Rules implemented</h2><ol><li><strong>ENSO surprise:</strong> the newly available ONI observation minus a rolling AR(1) expectation, standardized only with earlier errors. The observation enters after a three-month publication delay.</li><li><strong>Cooling-load forecast:</strong> disclosed-capacity-weighted monthly cooling-degree proxy across Illinois, Indiana, Iowa, Michigan, North Dakota and Ohio. A calendar-month regression relates excess cooling load to ONI known three months earlier.</li><li><strong>Fundamental confirmation:</strong> only Massive articles discussing orders, backlog, guidance, capacity, sales, data centers or cooling enter. Ticker-level sentiment is aggregated before the trade month.</li><li><strong>Relative momentum:</strong> the stock's trailing three-month return less XLI, standardized across the universe.</li><li><strong>Selection:</strong> rank = 35% news confirmation + 45% relative momentum + 20% previously validated cooling-exposure purity. Select at most two confirmed stocks. Exclude SPXC because its prior research classification was below the 0.40 purity threshold.</li><li><strong>Risk control:</strong> inverse six-month volatility weights. Full gross exposure in a positive XLI regime, half otherwise, with a further 25% reduction when cooling-load confirmation is absent.</li><li><strong>Hedge:</strong> hedge half the estimated 24-month XLI beta when the XLI trend is positive and all of it when the trend is weak. Charge 15 bp per one-way unit of stock and hedge turnover.</li></ol>
<div class='formula'>Rstrategy,t = Σ wᵢ,t Rᵢ,t − hₜ(Σ wᵢ,t βᵢ,XLI) RXLI,t − 0.0015 × turnoverₜ; hₜ = 0.5 in a positive XLI regime, otherwise 1.0</div></section>
<section><h2>Forecast model quality</h2><p>For prices, a ridge model explains each company's monthly return in excess of XLI using metals, ENSO surprise, cooling forecast, news and momentum. Market, metal and cross-company residual blocks are resampled 20,000 times per scenario. Negative blocked-CV R² means the model is worse than the fold mean and its scenario forecast should receive little weight.</p>{diag_table}</section>
<section><h2>What would invalidate the trade</h2><ul><li>The 8 October NOAA update does not raise the forecast relative to information already available on 11 September.</li><li>Forecast warming remains concentrated in cold states and modeled cooling-degree demand stays near zero.</li><li>No supplier raises data-center backlog, orders, revenue guidance or capacity utilization.</li><li>The candidate underperforms XLI and the XLI trend turns negative.</li><li>Expected beta-hedged return is smaller than transaction costs and the model's residual range.</li></ul></section>
<section><h2>Sources and files</h2><ul><li><a href='https://massive.com/docs/rest/stocks/overview'>Massive stock and news APIs</a>.</li><li><a href='https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/'>NOAA CPC ONI</a>.</li><li><a href='../enso_forecast_2026/ENSO_forecast_September_December_2026.html'>Prior ONI and state-temperature forecast</a>.</li><li><a href='../datacenter_cooling_monte_carlo_2026/datacenter_cooling_stock_monte_carlo_oct_dec_2026.html'>Prior 27-scenario cooling-stock simulation</a>.</li><li><a href='calculated/backtest_monthly.csv'>Monthly backtest data</a> and <a href='calculated/forecast_scenarios.csv'>forecast scenario data</a>.</li></ul></section>
<footer>Research simulation, not investment, tax or legal advice. Forecast distributions are conditional on the stated assumptions and do not represent guaranteed outcomes.</footer></main></body></html>"""


def main() -> None:
    CALC.mkdir(parents=True, exist_ok=True)
    prices, log_returns, simple_returns, last_prices = load_data()
    oni_metals = load_oni_metals()
    signals, news, relmom = build_signal_panel(prices, log_returns, oni_metals)
    bt, positions = backtest(prices, simple_returns, signals, news, relmom)
    perf = performance(bt)
    models, diagnostics = fit_forecast_models(log_returns, oni_metals, signals, news, relmom)
    forecast, portfolio, assumptions = forecast_scenarios(prices, log_returns, oni_metals, signals, news, models, last_prices)
    bt.to_csv(CALC / "backtest_monthly.csv", index=False)
    positions.to_csv(CALC / "backtest_positions.csv", index=False)
    perf.to_csv(CALC / "backtest_summary.csv", index=False)
    forecast.to_csv(CALC / "forecast_scenarios.csv", index=False)
    portfolio.to_csv(CALC / "forecast_strategy.csv", index=False)
    assumptions.to_csv(CALC / "scenario_assumptions.csv", index=False)
    diagnostics.to_csv(CALC / "forecast_model_diagnostics.csv", index=False)
    inputs = list(PRICE_ROOT.glob("massive_*_monthly.json")) + list(RAW_NEWS.glob("*.json")) + [METALS_PATH, TEMP_PATH, FORECAST_PATH, STATE_EFFECT_PATH]
    provenance = {"generated": "2026-09-13", "information_date": "2026-09-11", "seed": SEED,
                  "paths_per_scenario": N_PATHS, "transaction_cost": COST,
                  "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}}
    (CALC / "provenance.json").write_text(json.dumps(provenance, indent=2))
    REPORT.write_text(report_html(bt, positions, perf, forecast, portfolio, assumptions, diagnostics))
    print(perf.to_string(index=False))
    print("\nForecast strategy\n", portfolio.to_string(index=False))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__": main()
