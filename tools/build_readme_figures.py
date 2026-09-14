"""Build the static figures embedded in the repository-level README.

The script reads only saved, calculated CSV outputs.  It does not contact any
market-data or climate-data API.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324d"
BLUE = "#2878b5"
TEAL = "#2a9d8f"
ORANGE = "#e07a3f"
RED = "#c44e52"
GREY = "#6b7280"
LIGHT = "#d9e2ec"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#334155",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d7dee7",
            "grid.alpha": 0.75,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def correlation_candidates() -> None:
    df = pd.read_csv(ROOT / "strategy_correlation_comparison" / "raw_correlation_candidates.csv")
    df = df.sort_values("r")
    labels = (
        df["relationship"]
        .str.replace("(1m target)", "", regex=False)
        .str.replace(" minus ", " − ", regex=False)
    )
    y = np.arange(len(df))
    left = df["r"] - df["ci_low"]
    right = df["ci_high"] - df["r"]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    colors = [ORANGE if "energy" in x.lower() else BLUE for x in df["family"]]
    ax.errorbar(
        df["r"],
        y,
        xerr=np.vstack([left, right]),
        fmt="none",
        ecolor=LIGHT,
        elinewidth=5,
        capsize=0,
        zorder=1,
    )
    ax.scatter(df["r"], y, c=colors, s=55, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Raw Pearson correlation r (95% Fisher interval)")
    ax.set_title("Best descriptive relationships are mostly negative and modest")
    ax.grid(axis="x")
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.text(row["ci_high"] + 0.01, yi, f"n={int(row['n'])}", va="center", fontsize=8, color=GREY)
    ax.text(
        0.01,
        -0.18,
        "Intervals describe sampling uncertainty only; they do not correct every series for selection or overlapping horizons.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GREY,
    )
    save(fig, "correlation_candidates.png")


def walkforward_transfer() -> None:
    df = pd.read_csv(ROOT / "enso_metals_walkforward" / "calculated" / "walkforward_summary.csv")
    keep = df[
        ((df["target_kind"] == "ONI → metal benchmark"))
        | ((df["asset"].isin(["MOD", "VRT", "JCI", "ETN"])) & (df["target_kind"] == "metal → infrastructure"))
    ].copy()
    keep["label"] = keep["asset"] + np.where(
        keep["target_kind"].eq("metal → infrastructure"), " (metal→stock)", " (ONI→metal)"
    )
    keep["family"] = np.where(keep["target_kind"].eq("ONI → metal benchmark"), "ONI→metal", "metal→stock")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for family, group, color in [
        ("ONI→metal", keep[keep["family"] == "ONI→metal"], BLUE),
        ("metal→stock", keep[keep["family"] == "metal→stock"], ORANGE),
    ]:
        ax.scatter(
            group["median_train_abs_r"],
            group["ic"],
            s=np.clip(group["n"], 15, 220),
            alpha=0.82,
            label=family,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["label"],
                (row["median_train_abs_r"], row["ic"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0, color=NAVY, linewidth=1)
    ax.set_xlabel("Median absolute training correlation selected by the model")
    ax.set_ylabel("Walk-forward information coefficient")
    ax.set_title("Descriptive correlation often failed to transfer out of sample")
    ax.grid(True)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        -0.18,
        "Marker area scales with out-of-sample months. VRT is visually prominent but has only 17 months and two frozen blocks.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GREY,
    )
    save(fig, "walkforward_transfer.png")


def risk_return_tradeoff() -> None:
    df = pd.read_csv(
        ROOT / "enso_cooling_confirmed_strategy_2026" / "calculated" / "risk_management_summary.csv"
    )
    df = df[df["variant"].isin(["basic", "inverse_vol", "vol_target", "trend", "drawdown", "tail_budget", "combined", "core_signal", "confirmed", "XLI"])].copy()
    df["drawdown_abs"] = -100 * df["max_drawdown"]
    df["cagr_pct"] = 100 * df["cagr"]
    colors = np.where(df["meets_portfolio_gate"], TEAL, np.where(df["variant"].eq("basic"), RED, BLUE))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.scatter(df["drawdown_abs"], df["cagr_pct"], c=colors, s=75, edgecolor="white", linewidth=0.8)
    offsets = {
        "basic": (6, 6),
        "inverse_vol": (6, -12),
        "vol_target": (6, -12),
        "trend": (6, 7),
        "drawdown": (6, 7),
        "tail_budget": (6, -12),
        "combined": (6, 7),
        "core_signal": (6, -12),
        "confirmed": (6, 7),
        "XLI": (6, 7),
    }
    for _, row in df.iterrows():
        ax.annotate(
            row["variant"].replace("_", " "),
            (row["drawdown_abs"], row["cagr_pct"]),
            xytext=offsets[row["variant"]],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Maximum drawdown magnitude (%) → worse")
    ax.set_ylabel("Backtested CAGR (%) → better")
    ax.set_title("Risk controls traded return for smaller drawdowns")
    ax.grid(True)
    selected = df.loc[df["variant"].eq("drawdown")].iloc[0]
    ax.scatter([selected["drawdown_abs"]], [selected["cagr_pct"]], facecolors="none", edgecolors=TEAL, s=190, linewidth=2)
    ax.text(
        0.01,
        -0.18,
        "The drawdown brake was the only variant passing the predeclared gate: ≥50% CAGR retention and ≥30% drawdown reduction, without leverage.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GREY,
    )
    save(fig, "risk_return_tradeoff.png")


def state_cooling_cost() -> None:
    df = pd.read_csv(
        ROOT / "enso_datacenter_cooling_2026" / "calculated" / "affected_states_cooling_spend.csv"
    ).dropna(subset=["capacity_gw_2025_disclosed"])
    df = df.sort_values("central_incremental_spend_usd_disclosed_capacity")
    values = df["central_incremental_spend_usd_disclosed_capacity"] / 1e6

    fig, ax = plt.subplots(figsize=(9.3, 5.2))
    bars = ax.barh(df["state"], values, color=TEAL, alpha=0.9)
    ax.bar_label(bars, labels=[f"${v:.2f}m" for v in values], padding=4, fontsize=9)
    ax.set_xlabel("Central incremental December cooling-electricity cost (USD millions)")
    ax.set_title("Physical bridge: about $9.19m across six capacity-disclosed states")
    ax.grid(axis="x")
    ax.set_xlim(0, max(values) * 1.25)
    ax.text(
        0.01,
        -0.19,
        "This is a one-month electricity-cost estimate, not equipment capex or supplier revenue. Central weather-driven equipment capex was estimated at $0.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GREY,
    )
    save(fig, "state_cooling_cost.png")


if __name__ == "__main__":
    setup()
    correlation_candidates()
    walkforward_transfer()
    risk_return_tradeoff()
    state_cooling_cost()
    print(f"Wrote README figures to {OUT}")
