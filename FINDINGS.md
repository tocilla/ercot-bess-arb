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

### 2026-04-24 — perfect-foresight LP ceiling, HB_NORTH 2011–2024

**Config:** default battery (100 MW / 200 MWh / η=0.85 / $2/MWh deg), 1 cycle/day,
`tz=US/Central`. Script: `scripts/run_baselines_real.py`. Solver: HIGHS.

**Data:** same as previous entry — HB_NORTH RTM SPP, 2011–2024, 490,943
intervals, 5,114 days.

**What:** added perfect-foresight LP dispatch as the **ceiling** (METHODOLOGY
§5.2). Per-day LP with SOC carried across days, total battery-side
throughput ≤ 2 × cycles_per_day × usable. Also added regime breakdown
(normal / scarcity-only / negative-only / both).

**How:** cvxpy + HIGHS, one LP per local day. 5,114 LPs solve in ~64s on
full history — about 12ms per day's LP. Throughput capped by the same
rule as the floor baseline for apples-to-apples comparison.

**Headline numbers:**

| Strategy                  | Total revenue | Mean $/day | Std $/day | Worst day | Best day   | Sharpe | % clipped |
|---------------------------|---------------|------------|-----------|-----------|-----------|--------|-----------|
| natural-spread floor      | \$73.22M      | \$14,317   | \$67,090  | -\$830,710| +\$1,498,801 | 0.21   | 2.85%     |
| perfect-foresight ceiling | **\$81.31M**  | \$15,900   | \$66,277  | **\$0**   | +\$1,489,748 | 0.24   | 0.99%     |

**Floor captures 90.0% of ceiling.** Headroom = \$8.1M over 14 years,
i.e. **\$5.8/kW-yr** on a 100 MW battery — the absolute maximum any
dispatch strategy can earn *above* the mechanical floor.

**Regime breakdown (where is the money actually earned):**

| Regime         | Days | % days | Floor revenue | % of total | Ceiling revenue | % of total |
|----------------|------|--------|---------------|------------|-----------------|------------|
| Scarcity only  | 383  | 7.5%   | \$48.30M      | **66.0%**  | \$53.13M        | **65.3%**  |
| Normal         | 4175 | 81.6%  | \$20.21M      | 27.6%      | \$23.18M        | 28.5%      |
| Negative only  | 536  | 10.5%  | \$3.24M       | 4.4%       | \$3.54M         | 4.4%       |
| Scarcity + neg | 20   | 0.4%   | \$1.46M       | 2.0%       | \$1.46M         | 1.8%       |

Scarcity is defined as `max_price_on_day > \$500`. Negative as
`min_price_on_day < 0`.

**What this means — revised ML plan:**

1. **The ML opportunity is narrow.** Floor already captures 90% of ceiling.
   Any forecasting model must beat the floor by a large fraction of the
   \$5.8/kW-yr headroom to be worth the complexity. Hunting for MAE
   improvements on normal days is fighting for scraps.

2. **Scarcity events are 90% of the game.** 7.5% of days supply 66% of
   revenue. A model that correctly *anticipates* scarcity — lets the
   battery sit at full SOC the day before a heat dome or system outage —
   captures most of the alpha. Models that fit well on normal-day noise
   will look great on MAE but do nothing for revenue.

3. **Floor's negative days are the clearest win.** Floor has worst day
   -\$830k; ceiling has worst day \$0 (LP idles on unprofitable days).
   A *skip-if-unprofitable* gate alone closes a large piece of the gap —
   no forecasting required, just a spread-vs-cost check using the daily
   price range we'd have anyway.

4. **Published "BESS ML" revenue numbers are largely spread capture.**
   Any paper that reports impressive \$/MWh-day without comparing to the
   natural-spread floor is overstating ML's contribution.

**What broke / what surprised me:**
- First LP formulation had a closure constraint (`soc_end = soc_start` per
  day) which made the LP *strictly more restrictive* than the floor
  baseline and caused LP < floor — confusing. Removed closure; added
  explicit cross-day SOC carry-over via a simulator state tracker. Now
  LP always ≥ floor as it should.
- Initial cycle-cap was charge-side only (`sum(eta·p_c·dt) ≤ cycles·usable`).
  With initial SOC at 50%, the LP could drain initial SOC for free on day
  1 without counting against the cycle. Changed to total battery-side
  throughput cap (`sum(eta·p_c + p_d/eta)·dt ≤ 2·cycles·usable`),
  matching the physical definition of "1 cycle = once in + once out".
- Floor's clipping rate (2.85%) is meaningfully higher than ceiling's
  (0.99%). The baseline's blind cycling occasionally hits SOC limits the
  LP avoids.

**Next:**
1. Regime-stratified baselines — how does floor perform if we add a
   *skip-if-unprofitable* gate? Probably recovers most of the ceiling's
   advantage, setting a tougher bar for ML.
2. First forecasting baselines (persistence, seasonal-naive).
3. Walk-forward evaluation harness.
4. First ML model (LightGBM with lags + calendar), scored against floor
   and headroom-capture not just MAE.

---

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
