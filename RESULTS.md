# Results — final, one-shot test reveal

Test window: **2023-01-01 → 2024-12-31** (731 days, HB_NORTH ERCOT RTM SPPs).
Battery spec: 100 MW / 200 MWh / 85% round-trip / 5–95% SOC / $2/MWh degradation
/ 1 cycle/day cap. Per-strategy detail in [scripts/test_reveal.py](scripts/test_reveal.py)
and [results/test_reveal.csv](results/test_reveal.csv).

## Headline

**Deployable claim:** the locked-spec ML ensemble adds **+$5.29M revenue
over the 2-year test window** vs. a persistence baseline — equivalent to
**+$2.6M/year** of arbitrage value on a 100 MW battery, or **+26.3 pp** of
ceiling captured over the most relevant deployable alternative.

**Academic-comparable framing:** the same ensemble captures **77.13%** of
the perfect-foresight revenue ceiling. Within the academic literature's
typical 50–70% range for 1-cycle/day arbitrage on daily-vintage features;
below commercial production claims (Ascend Analytics: 90–95%) which use
full-stack DAM + RTM + ancillary-services joint optimization with real-time
data feeds — a deliberately wider scope than this project.

| Strategy | Revenue | $/kW-yr | Δ vs persistence | % of ceiling | Forecast MAE |
|---|---:|---:|---:|---:|---:|
| do nothing | $0 | 0.0 | −$10.24M | 0.00% | — |
| fixed time of day (charge 3 AM, discharge 5 PM) | $3,103,800 | 15.5 | −$7.14M | 15.42% | — |
| persistence forecast (yesterday's prices) | $10,243,787 | 51.2 | (baseline) | 50.88% | $33.21 |
| seasonal-naive forecast (4-week same-DOW median) | $10,595,535 | 52.9 | +$0.35M | 52.63% | $27.13 |
| **our ML ensemble (locked spec)** | **$15,529,439** | **77.6** | **+$5.29M** | **77.13%** | **$12.27** |
| natural-spread floor (*oracle, not deployable*) | $19,116,197 | 95.5 | +$8.87M | 94.95% | — |
| LP ceiling (*perfect foresight, theoretical*) | $20,133,631 | 100.6 | +$9.89M | 100.00% | — |

> **Why the bottom two rows aren't strategies.** The natural-spread floor
> and the LP ceiling both **cheat** by using realized prices that wouldn't
> have been knowable at decision time:
>
> - The **natural-spread floor** sorts each day's *realized* LMPs after the
>   fact, charges during the cheapest intervals, discharges during the most
>   expensive. Pure post-hoc oracle on the daily price ranks. A real
>   forecaster running threshold-rule dispatch can approach this number
>   but mathematically cannot exceed it.
> - The **LP ceiling** solves a perfect-foresight linear program given the
>   full sequence of true future prices. It's the absolute mathematical
>   maximum a battery with our spec could earn — by construction, no
>   forecasting strategy can ever beat it.
>
> Both are **benchmarks**, not algorithms. They tell us how big the
> arbitrage pie is (ceiling) and how much of it is theoretically reachable
> with perfect knowledge of the daily shape (floor). The only deployable
> comparison in this table is **persistence** — that's why the headline
> `Δ vs persistence` claim is measured against it.

Why two framings: the "% of ceiling" number is what the BESS literature reports
and how this project compares to published work. The "Δ vs persistence" number
is what answers "should I deploy this?" — it isolates what the model contributes
above what's already free in regular daily price structure (which persistence
already captures by accident). Both are valid; report both.

## Stability across seeds

| Metric | Mean | Std | Range |
|---|---:|---:|---:|
| % of ceiling | 75.99% | **0.43 pp** | 75.57 – 76.65% |
| MAE | $12.38 | $0.14 | $12.18 – $12.52 |

The 5-seed ensemble at 77.13% sits **+1.14 pp above the individual-seed
mean** — the same kind of variance-reduction lift we measured on val.

Per-seed std on test (0.43 pp) is much tighter than on val (2.84 pp);
2023–24 is a less volatile period than the val window 2020-11–2022-12,
which contained Winter Storm Elliott (Dec 2022) and most of 2021's
post-Uri repricing.

## Where the ML gap to floor lives — by regime

| Regime | n_days | Floor revenue | ML revenue | ML/Floor | Floor−ML gap |
|---|---:|---:|---:|---:|---:|
| scarcity_only (max > $500) | 63 | $14,176,916 | $11,396,000 | **80.4%** | $2.78M |
| normal | 542 | $3,958,754 | $3,300,605 | 83.4% | $0.66M |
| negative_only (min < $0) | 124 | $854,074 | $734,844 | 86.0% | $0.12M |
| both | 2 | $126,452 | $97,990 | 77.5% | $28k |

Scarcity-day capture is the limiting factor — same finding as on val,
but stronger here (80% of floor's scarcity revenue captured on test
vs. 69% on val). 78% of the total floor → ML gap on test still lives
on scarcity days.

## Comparison to industry / academic context

| Reference | % of perfect foresight |
|---|---:|
| Commercial production (Ascend Analytics SmartBidder, claimed) | 90–95% |
| Recent academic SOTA (1MW/1MWh CID battery, [arXiv 2501.07121](https://arxiv.org/abs/2501.07121)) | 89% |
| Typical academic forecast-driven results, 1-cycle/day arb | 50–70% |
| **This project on test set** | **77.13%** |
| Persistence baseline (this project) | 50.88% |

This project's 77.13% is **above the typical academic range** for daily-
vintage forecast-driven 1-cycle-per-day arbitrage and **below commercial
production claims**. The gap to commercial is largely scope: production
optimizers use real-time data feeds, dynamic intraday re-bidding, and
joint DAM + RTM + ancillary-services optimization — capabilities deliberately
outside this project's bounds.

## What this means in practice

1. **Beating persistence by +26 pp** is the productizable claim. On a 100 MW
   battery, that's ~$2.6M/year of additional arbitrage revenue vs. running
   the same dispatch logic on a "yesterday's prices" forecast. Real, robust
   across seeds, holds out-of-sample.
2. **Below the natural-spread floor by 17.8 pp.** That floor is a *theoretical*
   benchmark — it requires knowing realized prices in advance. It's a useful
   yardstick for "how close can perfect forecasting get?" but not a deployable
   baseline.
3. **Scarcity-day timing is the remaining ML headroom.** ~80% of the residual
   gap to floor is on the 8.6% of days with peak prices > $500. Closing that
   gap likely requires real-time generator-outage information and intraday-
   vintage forecasts that our daily-vintage feature set doesn't include.

## Locked-in model spec (what produced the 77.13%)

| Component | Value |
|---|---|
| Algorithm | LightGBM, quantile objective at α=0.5 (median) |
| Iterations | 200 |
| Hyperparameters | Defaults (no tuning on val) |
| Features (27) | 7 price lags, 3 rolling-price stats, 5 calendar, 5 load lags + rolling, 7 EIA-930 (demand actual + forecast + forecast error + wind/solar lags) |
| Training window | 2019-01-01 → boundary (truncated; full-history NaN-fill measured worse on val) |
| Walk-forward | Refit every 30 days |
| Seeds | 7, 13, 23, 42, 101 (predictions averaged) |
| Dispatch | Same-day forecast-gated threshold rule, 1 cycle/day cap |

Methodology details: [METHODOLOGY.md](METHODOLOGY.md). Negative results that
shaped the spec: [FINDINGS.md](FINDINGS.md).

## What we did NOT touch

- **No hyperparameter tuning** on val. LightGBM defaults throughout.
- **No model selection on test results.** Spec was frozen before reveal.
- **Test set was touched once.** Per [METHODOLOGY.md](METHODOLOGY.md) §1.
