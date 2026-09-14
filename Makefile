SHELL := /bin/sh
PYTHON ?= .venv/bin/python
MPLCONFIGDIR ?= /tmp/enso-research-matplotlib
export MPLCONFIGDIR

.DEFAULT_GOAL := help
.NOTPARALLEL: full-offline

.PHONY: help setup full-offline forecast insurance-ag metals energy compare \
	cop-xle options cooling-spend cooling-capex cooling-mc cooling-news \
	confirmed risk straddles insurance-event insurance-peak figures test-all

help:
	@echo "Offline ENSO research targets"
	@echo "  make setup             Create .venv and install tested dependencies"
	@echo "  make full-offline      Rebuild every analysis from cached/local inputs"
	@echo "  make test-all          Run all repository checks"
	@echo "  make forecast          Seasonal ENSO forecast and report"
	@echo "  make insurance-ag      Insurance/agriculture study"
	@echo "  make metals            ENSO/metals/cooling walk-forward study"
	@echo "  make energy            El Nino energy-equity study"
	@echo "  make compare           Cross-strategy raw-correlation comparison"
	@echo "  make cop-xle           COP/XLE strong-event walk-forward study"
	@echo "  make options           COP/XLE options adaptation"
	@echo "  make cooling-spend     State cooling-spend estimate"
	@echo "  make cooling-capex     Cooling capex and supplier screen"
	@echo "  make cooling-mc        Cooling-stock Monte Carlo"
	@echo "  make cooling-news      Cached-news priced-in diagnostic"
	@echo "  make confirmed         Confirmed cooling-equity strategy"
	@echo "  make risk              Cooling-strategy risk overlays"
	@echo "  make straddles         Infrastructure long-straddle backtest"
	@echo "  make insurance-event   Insurance/ONI event study"
	@echo "  make insurance-peak    Peak-El-Nino insurer strategy"
	@echo "  make figures           Rebuild root README figures"

setup:
	python3.14 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

forecast:
	$(PYTHON) enso_forecast_2026/prepare.py
	$(PYTHON) enso_forecast_2026/model.py
	$(PYTHON) enso_forecast_2026/report.py

insurance-ag:
	$(PYTHON) enso_study/enso_study_analyze.py
	$(PYTHON) enso_study/enso_study_report.py

metals:
	$(PYTHON) enso_metals_walkforward/run_analysis.py

energy:
	$(PYTHON) enso_energy_peaks/run_analysis.py

compare: metals energy
	$(PYTHON) strategy_correlation_comparison/run_compare.py

cop-xle:
	$(PYTHON) enso_cop_xle_strong_event_walkforward/run_analysis.py

options: cop-xle
	$(PYTHON) enso_options_adaptation/run_analysis.py

cooling-spend: forecast
	$(PYTHON) enso_datacenter_cooling_2026/run_analysis.py

cooling-capex: cooling-spend
	$(PYTHON) enso_datacenter_cooling_capex_2026/run_analysis.py

cooling-mc: cooling-capex metals
	$(PYTHON) datacenter_cooling_monte_carlo_2026/run_analysis.py

cooling-news:
	$(PYTHON) datacenter_cooling_monte_carlo_2026/analyze_news_pricing.py

confirmed: cooling-mc
	$(PYTHON) enso_cooling_confirmed_strategy_2026/run_analysis.py

risk: confirmed
	$(PYTHON) enso_cooling_confirmed_strategy_2026/risk_management_analysis.py

straddles:
	$(PYTHON) dc_infrastructure_straddles_2026/run_analysis.py

insurance-event:
	$(PYTHON) insurance_oni_event_study/run_analysis.py

insurance-peak: insurance-event
	$(PYTHON) insurance_peak_strategy/run_strategy.py

figures: compare risk cooling-spend
	$(PYTHON) tools/build_readme_figures.py

full-offline: forecast insurance-ag compare options cooling-news risk straddles insurance-peak figures
	@echo "Offline rebuild complete. Open README.md and the project HTML reports."

test-all:
	$(PYTHON) tools/test_history.py
	$(PYTHON) enso_study/test_enso_study.py
	$(PYTHON) enso_forecast_2026/test_forecast.py
	$(PYTHON) enso_metals_walkforward/test_analysis.py
	$(PYTHON) enso_energy_peaks/test_analysis.py
	$(PYTHON) enso_cop_xle_strong_event_walkforward/test_analysis.py
	$(PYTHON) enso_options_adaptation/test_analysis.py
	$(PYTHON) enso_datacenter_cooling_2026/test_analysis.py
	$(PYTHON) enso_datacenter_cooling_capex_2026/test_analysis.py
	$(PYTHON) datacenter_cooling_monte_carlo_2026/test_analysis.py
	$(PYTHON) enso_cooling_confirmed_strategy_2026/test_analysis.py
	$(PYTHON) enso_cooling_confirmed_strategy_2026/test_risk_management.py
	$(PYTHON) dc_infrastructure_straddles_2026/test_analysis.py
	$(PYTHON) insurance_oni_event_study/test_analysis.py
	$(PYTHON) insurance_peak_strategy/test_strategy.py
