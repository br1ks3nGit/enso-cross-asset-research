# ENSO cooling-equity strategy and risk-management study

This directory contains the complete research implementation for the data-center
cooling-equity strategy. It includes the original point-in-time ENSO signal, the
October–December 2026 scenario simulation, and a new comparison of portfolio risk
controls.

The most important risk result is not that the most defensive rule had the smallest
loss. The useful comparison is how much basic-basket return was retained for each
unit of drawdown reduction. On the June 2021–August 2026 sample, the fixed drawdown
brake was the only tested overlay that satisfied the predeclared gate of retaining at
least 50% of the basic basket's CAGR while reducing maximum drawdown by at least 30%
and using no leverage. This is an in-sample policy comparison over a short, favorable
period for data-center equities; it is not a forecast of future performance.

## Reproduce without any API call

The risk-management work uses only files already present on disk. It does not read an
API key and does not contact Massive.

```bash
.venv/bin/python enso_cooling_confirmed_strategy_2026/risk_management_analysis.py
.venv/bin/python enso_cooling_confirmed_strategy_2026/test_risk_management.py
```

To reproduce the previously generated signal backtest and scenario report from the
same local inputs:

```bash
.venv/bin/python enso_cooling_confirmed_strategy_2026/run_analysis.py
.venv/bin/python enso_cooling_confirmed_strategy_2026/test_analysis.py
```

`download_massive_news.py` is retained solely to document how the local news archive
was originally assembled. It is not called by either analysis script and should not
be run for this study.

## Main deliverables

- `cooling_strategy_risk_management_report.html` — risk-policy comparison, stress
  months, formulas, recommendation and limitations.
- `confirmed_enso_cooling_strategy_report.html` — original signal backtest and
  October–December 2026 scenarios.
- `calculated/risk_management_monthly.csv` — one net return per policy and month,
  with each prior-data risk state.
- `calculated/risk_management_positions.csv` — stock weights, hedge, turnover and
  transaction cost for each policy and month.
- `calculated/risk_management_summary.csv` — performance, tail-risk, benchmark and
  return-retention metrics.
- `calculated/risk_management_next_allocation.csv` — September 2026 policy weights
  frozen from complete data through August; it is an audit record, not a late entry
  instruction.
- `calculated/risk_policy.json` — immutable policy constants used in the test.
- `calculated/backtest_monthly.csv` and `backtest_positions.csv` — original confirmed
  signal returns and positions.
- `calculated/forecast_scenarios.csv`, `forecast_strategy.csv`,
  `scenario_assumptions.csv` and `forecast_model_diagnostics.csv` — scenario outputs.
- `calculated/provenance.json` — hashes of the original analysis inputs.

## Data and universe

The cooling universe is `MOD`, `AAON`, `VRT`, `NVT`, `JCI`, `TT` and `SPXC`; `XLI` is
the industrial-equity benchmark and hedge instrument. Local JSON files named
`massive_<ticker>_monthly.json` in the adjacent
`datacenter_cooling_monte_carlo_2026/raw` directory contain previously downloaded
adjusted monthly bars. The analysis never sends an order.

The original signal also reads the local ONI/metals CSV, state-temperature JSONL,
state ENSO effects, supplier classification and previously downloaded Massive news
JSON. The full confirmation backtest begins in June 2021 because the local news
archive is not usable for most names before then. The test ends in August 2026 so a
complete monthly return follows every information date.

The company list is a research universe chosen with present knowledge. It therefore
has survivorship and selection bias. Adjusted bars reduce split/dividend errors but do
not reconstruct delisted securities, historical index membership, bid–ask spreads or
execution at a particular time of day.

## Code architecture: original cooling strategy

### `download_massive_news.py`

- `fetch` paginates a ticker/year news query, removes the key from each stored next
  URL and returns the records plus request count. The local files contain article
  metadata and ticker-level sentiment.
- `main` reads the key from the adjacent configuration only when the downloader is
  deliberately run, loops over tickers and years, deduplicates articles and writes
  local JSON. The key is never written to an output. This script is not used by the
  present risk analysis.

### `run_analysis.py`: data and signals

- `load_monthly` converts each local aggregate timestamp to a monthly period and
  returns adjusted close and volume series.
- `load_data` aligns the seven stocks and XLI, then calculates log and simple returns.
  Simple returns drive portfolio P&L; log returns are used by the forecasting model.
- `load_oni_metals` loads ONI and copper, aluminum and nickel. Metal prices become
  monthly log returns so a persistent nominal price level does not create a spurious
  relationship.
- `oni_surprises` fits a rolling AR(1) model to ONI history available before each
  observation. The residual is standardized using earlier residuals. Its trade month
  is shifted forward three months, imposing a historical publication delay. This
  reduces look-ahead but does not reconstruct true NOAA vintages, which remains an
  important limitation.
- `capacity_cdd` converts state temperature to a cooling-degree proxy above 18°C and
  averages it using disclosed data-center capacity in Illinois, Indiana, Iowa,
  Michigan, North Dakota and Ohio.
- `cooling_forecast_series` estimates a separate calendar-month relation between
  excess cooling degree demand and ONI known three months earlier. A small fixed ridge
  penalty stabilizes sparse month-specific estimates. Only years before the target
  month enter each fit.
- `news_scores` keeps articles about orders, backlog, guidance, revenue, sales,
  earnings, capacity, contracts, data centers or cooling. Ticker sentiment is summed
  and divided by the square root of article count, then shifted one month so a test
  month cannot read its own news.
- `build_signal_panel` joins delayed ENSO data, the cooling estimate, lagged news and
  three-month stock return relative to XLI. The XLI regime is positive only when the
  prior close is above its trailing six-month mean and its trailing three-month log
  return is positive.
- `rolling_beta` estimates up to 24 prior monthly observations and clips each stock
  beta to 0.25–2.00. The clip limits hedge instability in short samples.

### `run_analysis.py`: portfolio and forecast

- `backtest` trades only when the delayed ONI level exceeds 0.5 and is rising or
  positively surprising. The cross-sectional rank is 35% lagged news, 45% relative
  momentum and 20% cooling-exposure purity. At most two confirmed stocks are held.
  Six-month inverse-volatility weights prevent the more volatile selected name from
  taking equal risk by default. Gross is 100% in a positive XLI regime and 50%
  otherwise, with another 25% reduction when modeled cooling demand is absent. The
  XLI hedge covers half estimated beta in a positive market regime and all estimated
  beta in a weak regime. A fixed 15 bp is charged for every one-way unit of stock and
  hedge turnover.
- `hac_mean_test` calculates a Newey–West-style mean-return test with three monthly
  lags. It helps identify serial-correlation-sensitive inference but cannot compensate
  for only eight active signal months.
- `performance` calculates total return, annualized return/volatility, Sharpe,
  drawdown, active hit rate and the HAC p-value for the confirmed signal, naive ONI
  basket, cooling basket and XLI.
- `fit_forecast_models` estimates ridge regressions for each stock's log return in
  excess of XLI. Inputs are XLI, three metal returns, ENSO surprise, modeled cooling,
  news and relative momentum. Four blocked time-series folds measure out-of-sample
  fit. Six of seven negative out-of-sample R² values are disclosed because scenario
  outputs must not be treated as precise targets.
- `ar1_metal_paths` fits fixed AR(1) models to each metal return. `sample_blocks`
  resamples contiguous residual blocks to preserve some short-run dependence.
- `current_cross_section` constructs the information-date news and relative-momentum
  ranks used for the October 2026 decision.
- `forecast_scenarios` creates 20,000 paths for Excellent, Normal and Conservative
  cases. It resamples market, metal and cross-company residual blocks. The Excellent
  case contains an explicit positive repricing assumption because that scenario was
  defined as cooling spending raising company prices; Normal does not add that
  uplift, and Conservative keeps the climate sleeve in cash.
- `wealth_svg`, `table` and `report_html` create a self-contained HTML file without
  JavaScript or external chart dependencies. `main` writes every calculated table,
  report and provenance hash.

## Code architecture: risk management

### `risk_management_analysis.py`

- `capped_inverse_vol` uses only the prior 12 monthly returns. Raw weights are
  proportional to `1 / volatility`; a water-filling loop redistributes weight above
  the 25% name limit. The cap limits concentration while preserving a fully invested
  long allocation.
- `trailing_risk_controls` calculates three independent exposure limits. The
  volatility estimate uses 12 prior shadow-core returns; tail loss uses the mean of
  the worst 10% of the prior 24; drawdown uses the shadow core's prior wealth. A
  shadow portfolio remains observable even while the live policy is de-risked, so the
  drawdown rule can re-enter after recovery.
- `simulate` builds seven portfolio variants month by month. It computes weights and
  all risk states before reading the test month's return. It charges 15 bp per unit of
  long plus hedge turnover. The fixed 85/15 allocation combines the risk-managed
  basket core with the pre-existing confirmed ENSO return series; the mix was not
  optimized.
- `performance_summary` calculates compound return, CAGR, volatility, Sharpe,
  Sortino, Calmar, maximum drawdown and duration, empirical VaR/expected shortfall,
  worst month/three months, XLI beta/correlation, upside/downside capture, turnover,
  cost and HKD capital results. It also applies the predeclared portfolio gate.
- `line_svg` draws capital and drawdown histories; `scatter_svg` displays annualized
  return against maximum drawdown; `stress_table` lists the basic basket's eight worst
  months; `report_html` assembles the self-contained report.
- `main` writes CSV/JSON outputs and the HTML report. No network function exists in
  this module.

## Risk policies and financial rationale

All thresholds were stated in code before evaluating the outputs. They are policy
choices, not fitted coefficients.

1. **25% name cap.** A seven-stock thematic basket is not economically diversified:
   several firms share data-center demand and industrial-cycle exposure. The cap
   prevents a low trailing-volatility estimate from concentrating more than one
   quarter of long capital in one issuer.
2. **Inverse volatility.** Equal dollars can produce unequal risk when stock
   volatility differs greatly. Inverse-volatility weighting reduces exposure to the
   more volatile names without requiring expected-return forecasts, which are weak in
   this small sample.
3. **20% annual volatility target, no leverage.** The target is below the basic
   basket's observed volatility. Exposure equals `min(1, 20% / trailing volatility)`;
   calm periods never cause borrowing. A no-leverage rule avoids amplifying errors in
   a short volatility estimate.
4. **XLI trend scaling.** Cooling shares retain broad industrial and equity beta. If
   prior XLI is below its six-month mean or lacks positive three-month performance,
   gross falls to 50%. This is a simple, observable regime rule rather than a return
   forecast.
5. **Drawdown brake.** Gross is 100% above a −10% shadow drawdown, 50% from −10% to
   −20%, and 25% below −20%. It limits further capital at risk after a sustained loss
   while maintaining some participation. The shadow series, not de-risked account
   wealth, determines recovery and prevents a permanent cash lock.
6. **Expected-shortfall budget.** If the worst 10% of the prior 24 shadow returns has
   an average loss larger than 8%, gross is reduced in proportion. Expected shortfall
   is used because it includes the size of tail observations rather than only a
   percentile cutoff. Twenty-four months is still a small tail sample, so the output
   is treated as a rough budget.
7. **Partial XLI beta hedge.** The combined core hedges 25% of estimated beta in a
   positive regime and 50% in a weak one. Full hedging would remove much of the broad
   equity return that the basket is meant to retain and also raises turnover and
   financing needs.
8. **Most-restrictive combined limit.** The combined core uses the minimum of the
   volatility, tail, drawdown and trend multipliers. It does not multiply the four
   limits, which could turn several moderate warnings into an unintended near-zero
   allocation.
9. **85/15 core–signal basket.** Eighty-five percent is assigned to the risk-managed
   diversified core and 15% to the sparse confirmed ENSO sleeve. This makes ENSO a
   satellite source of differentiated exposure rather than the whole portfolio. The
   allocation is fixed and not selected from a weight sweep.
10. **Transaction costs.** Every monthly change in a long or hedge weight is charged
    15 bp one way. This is deliberately more conservative than reporting frictionless
    rebalancing, but it still omits actual spreads, market impact, XLI borrow cost and
    tax.

## Tested policies

- Basic equal-weight cooling basket.
- Capped inverse-volatility core.
- Inverse-volatility core with only the 20% volatility target.
- Inverse-volatility core with only XLI trend scaling.
- Inverse-volatility core with only the shadow drawdown brake.
- Inverse-volatility core with only the expected-shortfall budget.
- Combined core with all four exposure limits and the partial XLI hedge.
- Fixed 85/15 combined-core/confirmed-signal portfolio.
- Existing confirmed ENSO signal and XLI as context.

The predeclared gate requires at least 50% CAGR retention, at least 30% maximum-
drawdown reduction, and no gross leverage. This gate is intentionally simple. A
different investor may prefer a stricter loss budget, but changing it after seeing the
results would create selection bias.

## Metric formulas

For monthly net return `r_t` and `N` observations:

- Wealth: `W_t = product(1 + r_i)` through month `t`.
- Total return: `W_N - 1`.
- CAGR: `W_N^(12/N) - 1`.
- Annual volatility: `sample_std(r) * sqrt(12)`.
- Sharpe with zero cash rate: `mean(r) / sample_std(r) * sqrt(12)`.
- Sortino with zero target: `mean(r) / sample_std(r where r < 0) * sqrt(12)`.
- Drawdown: `W_t / max(W_0 ... W_t) - 1`; maximum drawdown is its minimum.
- Calmar: `CAGR / abs(maximum drawdown)`.
- 95% monthly VaR: positive magnitude of the empirical 5th-percentile return.
- 95% monthly expected shortfall: positive magnitude of the mean return at or below
  that percentile.
- XLI beta: `covariance(portfolio, XLI) / variance(XLI)`.
- Upside/downside capture: mean portfolio return divided by mean XLI return in months
  when XLI is positive/negative.
- CAGR retention: `portfolio CAGR / basic-basket CAGR`.
- Drawdown reduction: `1 - abs(portfolio max drawdown) / abs(basic max drawdown)`.
- Turnover: sum of absolute month-to-month changes in all long and hedge weights.
- Net return: `sum(weight_i * stock_return_i) - hedge * XLI_return - 0.0015 * turnover`.

The zero cash rate keeps comparison consistent with the original report. It understates
the value of cash during high-rate periods and means reported Sharpe is not an excess-
over-T-bill Sharpe.

## Point-in-time controls

- Test-month stock returns never enter test-month weights.
- Volatility uses the prior 12 months; tail sizing uses the prior 24 months.
- The XLI regime uses prices through the prior month.
- Beta uses up to 24 prior months.
- The drawdown state uses prior shadow-core wealth.
- Original news and relative momentum are lagged one month.
- ONI is delayed three months. Revised ONI data are still used after the delay because
  true historical vintages are unavailable locally.
- The 85/15 sleeve allocation and every threshold are constants, not optimized grids.

## Important limitations

- Sixty-three months are insufficient to estimate extreme loss reliably. Empirical
  95% expected shortfall averages only a few observations.
- The period is dominated by a large data-center equity rally. CAGR and drawdown may
  change substantially in a flat or declining thematic cycle.
- One completed strong El Niño cannot validate the climate thesis independently of
  AI capital spending, interest rates and broad equity beta.
- Monthly bars hide intramonth gaps. The code does not claim to test ATR stops,
  trailing stops, options, limit orders, margin calls or execution timing.
- The XLI hedge requires a short position or an equivalent instrument. Borrow cost,
  distributions, margin and tracking differences are absent.
- Tax is excluded. Hong Kong personal tax treatment depends on residence, facts and
  whether activity is considered a trade; professional advice is required.
- Cash earns zero in the simulation. This is conservative for recent USD/HKD cash but
  historically inconsistent.
- Pairwise correlations, beta and volatility can rise together during a sell-off.
  Caps and scaling reduce exposure but cannot guarantee a loss ceiling.
- The analysis is research, not investment advice, and no output is a guaranteed
  return or price target.

## Verification

`test_analysis.py` checks the original backtest, forecast scenario ordering,
provenance and HTML structure. `test_risk_management.py` checks point-in-time output
length, complete returns, leverage/name caps, metric arithmetic, fixed policy values,
API-key absence and report structure. Tests are data-integrity checks, not evidence of
economic profitability.
