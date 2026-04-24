# Findings

Running log of experiment results, most recent first. Populated as experiments
complete. Every entry is self-contained: someone reading only that entry
should understand what was tried, what was measured, and what it means.

## Entry template

```
## YYYY-MM-DD — <short title>

**Config:** configs/<file>.yaml, commit <short SHA>
**Data:** <node(s)>, <date range>, split <train / val / test cutoffs>

**What:** one paragraph on what was run.

**How:** model class, features, walk-forward cadence, seeds.

**Results:**
| Strategy          | $/MWh/day | % of ceiling | MAE ($/MWh) | Sharpe | Notes |
|-------------------|-----------|--------------|-------------|--------|-------|
| Natural spread    |           |              | —           |        |       |
| Persistence       |           |              |             |        |       |
| Seasonal naive    |           |              |             |        |       |
| Classical + rule  |           |              |             |        |       |
| This run          |           |              |             |        |       |
| Perfect foresight |           | 100%         | 0           |        |       |

**Revenue attribution:**
- Spread: $__ / day
- Forecast skill: $__ / day
- Optimization: $__ / day

**What broke / what surprised me:** honest notes. Features that leaked, bugs
found, regimes where the model misbehaved, seeds that disagreed.

**Next:** concrete next action, referencing a milestone in [PLAN.md](PLAN.md).
```

---

## Log

<!-- Newest entry at the top. -->

### 2026-04-24 — natural-spread baseline, HB_NORTH 2011–2024

**Config:** default battery (100 MW / 200 MWh / η=0.85 / $2/MWh deg), 1 cycle/day,
`tz=US/Central`. Script: `scripts/run_oracle_real.py`. Commit: (TBD — this
entry will be amended after commit).

**Data:** ERCOT RTM SPP at HB_NORTH, 2011-01-01 → 2024-12-31 (UTC).
490,943 intervals (15-min), 5,114 days. Price: mean \$41.13, median \$22.18,
std \$286.85, min -\$251, max \$9,315. Negative-price intervals 0.98%,
scarcity (>\$500) 0.47%.

**What:** ran the natural-spread baseline (PLAN §6.1) — mechanical daily
cycle, charge in cheapest intervals, discharge in most expensive, using
realized prices (oracle on timing). No forecasting, no profitability gate.
This is the revenue floor.

**How:** `daily_oracle_schedule` → `run_dispatch`. Full 14-year pass
executed in <1 minute from cached parquet files.

**Headline numbers:**

| Strategy               | Total revenue | Mean \$/day | Std \$/day | Worst day   | Best day    | Sharpe (daily) | % intervals clipped |
|------------------------|---------------|-------------|------------|-------------|-------------|----------------|---------------------|
| idle (no dispatch)     | \$0           | \$0         | \$0        | \$0         | \$0         | —              | 0.00%               |
| natural-spread oracle  | **\$73.22M**  | \$14,317    | \$67,090   | -\$830,710  | +\$1,498,801 | 0.21           | 2.85%               |

Per \$/kW-year (100 MW battery): **\$52.3/kW-yr average**, range
\$19.5 (2015) → \$141.6 (2023).

**Revenue by year:**

| Year | Revenue       | Notes                                                |
|------|---------------|------------------------------------------------------|
| 2011 | \$7.47M       | Texas heat wave summer                               |
| 2012 | \$2.51M       | Cheap-gas era, low spreads                           |
| 2013 | \$2.39M       |                                                      |
| 2014 | \$3.50M       |                                                      |
| 2015 | \$1.95M       | **Lowest year**                                      |
| 2016 | \$1.97M       |                                                      |
| 2017 | \$1.87M       |                                                      |
| 2018 | \$4.04M       |                                                      |
| 2019 | \$9.00M       | Summer scarcity                                      |
| 2020 | \$3.08M       |                                                      |
| 2021 | \$7.96M       | Feb Uri storm: worst *interval* = -\$225k            |
| 2022 | \$8.37M       | Gas spike + Winter Storm Elliott                     |
| 2023 | \$14.16M      | **Highest year**, sustained volatility               |
| 2024 | \$4.95M       | Reversion                                            |

**What broke / what surprised me:**
- Data. ERCOT RTM SPP is 15-min, not 5-min as I initially had in DATA.md.
  Fixed: `src/data/ercot.py` is built around 15-min, dispatch sim uses
  `interval_hours=0.25`.
- Fetcher. One historical document-list endpoint hung >10 min on the first
  call for 2011. Killed and retried with different year order; subsequent
  calls were fast (server-side caching). Kept the per-year parquet cache so
  this is a one-time cost.
- Baseline volatility. The "natural-spread" floor isn't a smooth baseline
  — it swings from -\$830k days to +\$1.5M days. Daily Sharpe 0.21 is
  unattractive by trading standards; revenue is concentrated in tail days.
  Any ML model that reports mean-$/day without showing the distribution is
  obscuring the hard part of the problem.
- Regime dependence. Ratio of best to worst year is ~7.3×. Any test period
  of <3 years will be unrepresentative.
- Flat-day losses. The worst day (-\$830k, 2021-02-15 neighborhood) came
  from the baseline's mechanical "always cycle" rule caught against
  post-storm negative prices. This is a feature of this baseline as
  defined — it does not skip cycling on unprofitable days. A true perfect-
  foresight ceiling via LP will not have this property and will show a
  higher headroom than the naive cycle count suggests.

**Implications for the ML plan:**
- The natural-spread floor is big — \$5.23M/yr average, \$14.16M in 2023.
  Any forecasting model has to pay its way against this number, not
  against zero. The "spread component" of revenue attribution is going to
  dominate for years.
- Test on ≥ 3 contiguous years and run walk-forward. A single-year test is
  not a meaningful read.
- Scarcity events (>\$500) are 0.47% of intervals but supply a large
  fraction of revenue. Forecasting them is the real alpha source.
- Negative-price handling matters: 0.98% of intervals are negative, and
  the baseline sometimes buys at positive and has to sell at negative on
  regime-break days.

**Next:**
1. Add a true perfect-foresight *ceiling* (LP over each day) so we can
   report "% of ceiling captured" per METHODOLOGY §5.2.
2. Add regime-stratified metrics (normal / scarcity / negative-price days).
3. First real forecasting baseline (SARIMA or LightGBM with lag+calendar),
   evaluated under walk-forward.

---

### 2026-04-24 — project initialized

Repo scaffolded. No experiments run yet. First milestone is data pipeline
for one ISO, one node, RTM LMPs + load + renewables.
