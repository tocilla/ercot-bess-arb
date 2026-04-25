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

### 2026-04-25 — Pre-reveal: 5-seed ensemble lifts +5 pp; gap is concentrated on scarcity days

**Two cheap analyses on the session-best ML setup before test reveal.**

**(1) Seed ensemble.** Average the predictions from the 5 individual
seed runs we already had, dispatch on the averaged prediction.

| | Revenue | % ceiling |
|---|---|---|
| Individual seed mean | \$11.12M | 57.67 ± 2.84 pp |
| **5-seed ensemble**  | **\$12.08M** | **62.67%** |
| **Δ vs mean**        | **+\$963k** | **+5.00 pp** |

A +5 pp lift, roughly 2σ above the seed-stability mean. This is variance
reduction working as expected — averaging 5 stochastic predictors
collapses some of the per-seed dispatch-timing noise that the gate
amplifies. Costs nothing beyond the seed runs we'd already done.

**This is the new session-best on val: 62.67% of ceiling.** vs. floor
at 87.0%, gap = 24.3 pp.

**(2) Regime breakdown of the ensemble.** Where does the ML gap to
floor actually live?

| Regime | n_days | Floor revenue | Ensemble revenue | ML/Floor |
|---|---:|---:|---:|---:|
| **scarcity_only** (max > \$500) | 86 | **\$11,437,351** | \$7,897,262 | **69.0%** |
| normal | 588 | \$3,969,917 | \$3,109,396 | 78.3% |
| negative_only | 112 | \$937,436 | \$754,441 | 80.5% |
| both | 5 | \$421,955 | \$318,125 | 75.4% |

Floor → ML revenue gap, by regime:

| Regime | Floor − ML | % of total gap |
|---|---:|---:|
| scarcity_only | \$3,540,088 | **76.4%** |
| normal | \$860,521 | 18.6% |
| negative_only | \$182,995 | 4.0% |
| both | \$103,830 | 2.2% |

**76% of the entire floor-vs-ML gap lives on scarcity days alone.**
On the 86 scarcity days (11% of val), the ensemble only captures 69%
of the floor's revenue. On the other 705 days combined, it captures
~78–80% — relatively close.

This is the cleanest framing of the session's central finding:
- The natural-spread floor's strength is its perfect intra-day timing
  on the days where money is concentrated.
- ML's weakness is *also* its intra-day timing on those days. It gets
  the day's average shape roughly right (the 80% of normal days look
  fine), but the rank-ordering of intervals on scarcity days — exactly
  when prices spike from \$30 to \$3,000 within a few hours — is
  systematically off.
- This is consistent with every other diagnostic in the session: MAE
  and central-tendency point forecasts are not the limiting factor;
  intra-day rank prediction on tail days is.

**What this DOESN'T tell us:** whether revealing the test set will
generalize this or surprise us. The val window includes 86 scarcity
days; the test window may have a different number, with different
character (Winter Storm Elliott was Dec 2022, in val; the 2023 summer
heat dome falls in test; etc.). The 62.67% number is committed to as
the model spec; the test result will be whatever it is.

**Decision:** ready to reveal test. Locked-in spec:
- q50 LightGBM, 200 iters, default params
- 27 features (prices + load + EIA-930 demand+gen-mix)
- Truncated training, train_start = 2019-01-01
- Walk-forward retraining every 30 days
- 5 seeds (7, 13, 23, 42, 101), prediction averaging across seeds
- Forecast-gate dispatch
- 100 MW / 200 MWh / η=0.85 / 5–95% SOC / \$2/MWh degradation / 1 cycle/day

Test window per `configs/splits.yaml`: 2023-01-01 → 2024-12-31.

---

### 2026-04-25 — Decision-aware loss weighting: clearly hurts (and tells us why)

**Config:** Same model, same features, same val window, multi-seed (5).
Four loss-weighting schemes compared, all sharing the q50 quantile
objective and forecast-gate dispatch:

| Variant | Weight per training row |
|---|---|
| A_uniform | 1.0 (baseline) |
| B_devmean | \|y − daily_mean\| + 1 |
| C_topbotk | 5.0 if row is in day's top-8 or bottom-8, else 1.0 |
| D_pricemag | 1 + \|y\| / 100 |

**Per-seed:**

| Variant | mean % ceiling | std | mean MAE | Δ vs A | σ |
|---|---:|---:|---:|---:|---:|
| A_uniform | **57.67** | 2.84 pp | \$57.30 | — | — |
| B_devmean | 2.87 | 6.04 pp | \$3.8M (broken) | **−54.80** | **−12.3σ** |
| C_topbotk | 50.65 | 4.17 pp | \$55.17 | −7.03 | −2.0σ |
| D_pricemag | 44.41 | 7.26 pp | \$57.71 | −13.26 | −2.6σ |

**Every weighted variant lost decisively to uniform.** B_devmean was
catastrophically broken — predictions on middle-of-day hours went
wildly wrong (MAE in the millions on some seeds), revenue collapsed
to near zero. C and D were both about 2σ negative.

**The mechanism — and why this is the closing argument against
decision-focused learning too:**

The forecast-gate dispatch picks charge / discharge hours **by rank
within each day**, not by absolute price level. To get the rank right,
the model needs calibrated predictions across the entire daily
distribution. The middle hours don't trigger dispatch directly, but
they form the reference against which the extremes are ranked.

Upweighting extreme hours during training:
1. Pulls the model's predictions on the central distribution
   off-target (the loss is no longer telling it those rows matter).
2. The model becomes well-calibrated on the rare rows but poorly
   calibrated everywhere else.
3. The intra-day rank order breaks: a model that predicts \$80 for the
   true peak hour but \$70 for a middle hour (vs the previous \$60) is
   no better at picking the peak than uniform — and may be worse if
   the inflated middle hours rank above the true peak on a flat day.

**This generalizes to full decision-focused learning.** The differentiable
surrogate (soft argmax, top-k attention) likewise pushes gradient
signal toward the extremes. The result is the same kind of calibration
distortion. We've gotten 1σ of empirical evidence (weighted loss) plus
strong mechanistic reasoning that the rank-based nature of the dispatch
is hostile to any loss-shape that emphasizes extremes.

**Decision: this branch is exhausted.** Per the plan documented earlier:

> Step 1 if price-weighted MAE doesn't help → "DFL probably won't either"
> → go to Step 4 (stop and write up)

We're in Step 4. The remaining ML moves not yet tried — hourly-vintage
forecasts, real-time outage data — both require new data acquisition,
not new models. Daily-vintage forecast features (HRRR, ERCOT STWPF/STPPF)
already showed no lift. Loss shaping doesn't help. Models on the same
features won't help. The 30-pp gap from 57.67% (best ML on val) to
87.0% (floor) is the structural ceiling of what this problem framing,
with the data we have, can deliver.

**Final session-best ML on val:** 57.67 ± 2.84 pp of ceiling, q50 LGBM +
EIA-930 + 2019+ truncated training + uniform-weighted loss + forecast-
gate dispatch.

**Next:** reveal the test set on the session-best setup and on the
floor. One-shot, no further tuning.

---

### 2026-04-25 — Final: +ERCOT wind+solar forecasts. Variance shrinks 3×, mean unchanged.

**Config:** Same setup. Truncated training (2019+), forecast-gated
dispatch, multi-seed (7, 13, 23, 42, 101). Now compares:
- Baseline: prices + load + EIA-930 (27 features)
- +ERCOT STWPF + STPPF: adds wind + solar day-ahead vintaged forecasts
  (29 features)

Backfill complete: 1,461 wind docs + 1,461 solar docs across 2019-01
→ 2022-12 from ERCOT Public API at ~06 UTC publish.

**Per-seed:**

| Seed | Baseline % | +Wind+Solar % | Δ pp |
|------|-----------:|--------------:|-----:|
| 7    | 53.21      | 56.11         | +2.90 |
| 13   | 57.95      | 54.66         | −3.29 |
| 23   | 58.97      | 55.05         | −3.92 |
| 42   | 57.31      | 56.93         | −0.38 |
| 101  | 60.91      | 56.64         | −4.27 |

**Mean ± std:**

| Variant | Revenue | % ceiling | MAE |
|---|---|---|---|
| Baseline | \$11.12M ± \$548k | 57.67 ± **2.84 pp** | \$57.30 ± \$0.34 |
| +Wind+Solar | \$10.77M ± **\$191k** | 55.88 ± **0.99 pp** | \$57.31 ± \$0.39 |

**Delta: −1.79 pp of ceiling. Effect/noise: −0.94σ.** Within seed
noise — not statistically distinguishable from baseline.

**The interesting result is the variance, not the mean.**

The +wind+solar arm has roughly **one-third** the seed-to-seed
variance of baseline (\$191k vs \$548k revenue std; 0.99 vs 2.84 pp
of ceiling). The features are doing real work — the model is using
them, becomes more confident and seed-stable, point forecasts move
slightly. But none of that translates into more revenue.

This sharpens the diagnosis we've been circling all session:

> Dispatch revenue gains require correctly *timing* the small number
> of intervals on a small number of days that supply most of the
> revenue. Daily-resolution exogenous forecasts (HRRR temperatures,
> ERCOT STWPF/STPPF) help the model predict the *average* shape of
> price more reliably, but average-day predictions barely matter.
> The forecast-gate decides whether to cycle based on whole-day
> expected spread; the threshold dispatch picks intervals based on
> within-day price ordering. Neither benefits from "we expect average
> wind tomorrow" being well-calibrated.

**This is the closing experiment.** Per the plan documented yesterday,
the decision rule was:

> Δ < +2 pp or negative → forecasts don't help → go to Step 4 (stop
> and write up).

Δ = −1.79 pp, mean direction negative, 4/5 seeds individually negative.
This branch is exhausted with the data we have available.

**Final session-best ML result:**

  q50 LGBM + EIA-930 features + 2019+ truncated training + forecast-
  gate dispatch: **57.67 ± 2.84 pp of ceiling** on the validation
  window, vs. the natural-spread floor's 87.0%.

The remaining 30-pp gap to the floor (and 42-pp gap to the LP ceiling)
is **not closable with the daily-resolution exogenous-forecast features
we tested.** Closing it likely requires:

1. Decision-focused / end-to-end training (loss = dispatch regret,
   not MAE) — the only experiment we haven't tried that targets the
   MAE-vs-revenue gap directly.
2. Hourly-vintaged forecasts (24× more docs per endpoint) — possible
   but expensive, and the daily version showed no signal so probability
   of improvement is moderate.
3. Real-time generator-outage / system-condition data — likely the
   biggest gap, mostly requires market-participant credentials.

**Test set: still untouched.** Per METHODOLOGY §1.

---

### 2026-04-25 — Wind-only ERCOT STWPF as a feature: within noise, mildly negative

**Config:** Same setup as session-best. Truncated training (2019+),
forecast-gated dispatch. Compared two arms across 5 seeds (7, 13, 23,
42, 101):
- Baseline: prices + load + EIA-930 (27 features, the seed-stability
  reference)
- +ERCOT STWPF (wind): adds `ercot_stwpf_system_wide` from one daily
  vintaged ERCOT publish (publish hour ≈ 06 UTC, day-ahead horizon)

Solar (STPPF) **not yet included** — solar backfill still in progress
at run time. This is the wind-only intermediate.

**Per-seed results:**

| Seed | Baseline % | +STWPF % | Δ pp |
|------|-----------:|---------:|-----:|
| 7    | 53.21      | 56.56    | +3.35 |
| 13   | 57.95      | 55.44    | −2.51 |
| 23   | 58.97      | 54.05    | −4.92 |
| 42   | 57.31      | 55.92    | −1.39 |
| 101  | 60.91      | 48.81    | −12.10 |

**Mean ± std:**

| Variant | Revenue | % ceiling | MAE |
|---|---|---|---|
| Baseline | \$11.12M ± \$548k | **57.67 ± 2.84 pp** | \$57.30 ± \$0.34 |
| +STWPF | \$10.44M ± \$603k | **54.16 ± 3.13 pp** | \$57.29 ± \$0.36 |
| **Delta** | −\$680k | **−3.51 pp** | basically zero |

Effect/noise: **−1.18σ.** Below the 2σ "real-effect" threshold, but
trending negative — 4 of 5 seeds individually showed wind hurt, only
seed 7 helped. MAE didn't move.

**What this tells us — provisional:**

1. **Wind-forecast-only feature doesn't help and may mildly hurt.** Same
   pattern observed previously with daily-resolution HRRR temperature.
   A single value broadcast across 96 intraday intervals doesn't add
   timing information the model lacks from price lags + load + calendar.

2. **The seed dispersion grew slightly.** Adding a high-NaN feature
   (66.7% of training rows lack STWPF — pre-2019 history is missing)
   appears to add a touch more variance, which is consistent with the
   model occasionally over-relying on it.

3. **Don't conclude until +wind+solar is run.** Solar has a much
   stronger daily-cycle shape than wind (zero overnight, peak midday).
   It might be a more useful feature, or the combination might behave
   differently. Hold judgment.

**Decision logged for after solar backfill completes:**
- If +wind+solar mean lift ≥ 5 pp → keep, expand to hourly vintage.
- If +wind+solar lift < 2 pp or trending negative → confirms
  daily-resolution exogenous forecasts don't help in this framing.
  Move to "stop and write up" or try one targeted thing (e.g. ERCOT
  load forecast NP3-560-CD) before stopping.

---

### 2026-04-25 — Seed noise is big: ±2.8 pp of ceiling

**Config:** Session-best ML setup — q50 LGBM + EIA features + 2019+
truncated training + forecast-gate dispatch — run 5 times with seeds
{7, 13, 23, 42, 101} on the same val window.

**Results across 5 seeds:**

| Seed | Revenue     | % ceiling | MAE     |
|------|-------------|-----------|---------|
| 7    | \$10,255,694 | 53.21%    | \$57.77 |
| 13   | \$11,170,020 | 57.95%    | \$56.94 |
| 23   | \$11,367,643 | 58.97%    | \$57.48 |
| 42   | \$11,046,915 | 57.31%    | \$57.23 |
| 101  | \$11,740,928 | 60.91%    | \$57.06 |

**Statistics:**
- Revenue: mean \$11.12M ± \$548k std, range \$1.49M
- **% of ceiling: mean 57.67 ± 2.84 pp, range 7.71 pp**
- MAE: mean \$57.30 ± \$0.34 (extremely stable)

**Implications — a significant methodology correction:**

1. **HRRR's supposed −8.5 pp hurt on dispatch is within seed noise.**
   The HRRR vs EIA-only comparison was one-seed vs one-seed, and
   seed noise across 5 trials spans 7.7 pp. The "HRRR hurt dispatch"
   finding from the earlier entry is therefore not statistically
   reliable — it may be true, but we can't conclude it from a single
   seed. Re-running HRRR across the same 5 seeds is needed before
   the claim stands.

2. **Truncation's +5.8 pp win is near 2σ, probably real.** Full-history
   was 50.6%, truncated was 56.4%, delta 5.8 pp. Seed std is 2.84 pp,
   so the effect is roughly 2σ. Likely real but not definitive.

3. **MAE is a much more stable quality signal than revenue.** Seed std
   on MAE is <\$0.35; on revenue it's \$548k (5% of mean). This is a
   further argument that MAE doesn't reliably predict dispatch revenue
   — but it also means we can't use MAE as a noise-free tie-breaker
   for dispatch, because MAE is deaf to the intraday-timing errors
   that cost money.

4. **All future head-to-head comparisons should report mean ± std
   over ≥ 3 seeds.** The +ERCOT STWPF/STPPF experiment (backfill in
   progress) must be evaluated multi-seed, not single-seed, or we
   won't be able to distinguish signal from seed jitter.

**Revised session-best:** EIA-only baseline captures **57.67 ± 2.84 pp
of ceiling** on val, not 56.4%. 56.4% was the single seed we happened
to sample first.

**What broke / what surprised me:**
- I expected seed std around ±1 pp. Instead it's ±2.8 pp. That's
  enough to flip the sign of several findings in this session if they
  were re-run with a different seed.
- LGBM's seed sensitivity likely comes from bagging + feature sampling
  (bagging_fraction=0.9, feature_fraction=0.9). Lower those toward
  1.0 and seed noise would shrink — at the cost of giving up
  regularization. Not doing that now.

**Next:** resume ERCOT forecast backfill. When it completes, run
`scripts/run_ercot_forecasts_experiment.py` with **5 seeds** (not 1)
for both the baseline and the +STWPF/STPPF variant. Only conclude
"forecasts help" if the delta in means substantially exceeds 2.84 pp.

---

### 2026-04-25 — Adding HRRR improved MAE but HURT dispatch revenue

**Config:** Same as prior EIA experiment — q50 LGBM + forecast-gate + 2019+
truncated training. Val window 2020-11-01 → 2022-12-31. Added 3 HRRR
features on top of 27 prior features:
`hrrr_tx_mean_t2m_k`, `hrrr_tx_max_t2m_k`, `hrrr_tx_mean_wind10m_mps`.

Backfill scope: 1 cycle/day (12Z) × 1 forecast hour (F06, valid 18Z
= noon local). 1,457 HRRR rows across 2019–2022. Wall-clock: 30.8 min
with 8-way parallelism.

**Results (val window):**

| Variant | Features | Revenue | % ceiling | MAE | RMSE |
|---|---|---|---|---|---|
| Baseline (EIA only) | 27 | **\$10.86M** | **56.4%** | \$57.71 | \$611.09 |
| +HRRR | 30 | \$9.24M | 47.9% | **\$57.18** | **\$610.43** |

**HRRR hurt revenue by −\$1.63M (−8.5 pp of ceiling) while simultaneously
improving MAE and RMSE on the same val window.**

This is the session's *third* observation of the pattern "point-forecast
accuracy improvement ≠ dispatch revenue improvement":
1. Scarcity classifier (PR-AUC ↑, dispatch ↓ via rule override)
2. Combined scarcity-prob feature (MAE neutral, dispatch slightly worse)
3. HRRR daily temperature forecast (MAE ↓, dispatch ↓)

**Why this likely happens:**

- HRRR features are slow-moving daily aggregates. One 18Z temperature
  forecast applied to 96 intraday intervals gives no intraday-timing
  information the model didn't already have from `hour`, `dow`, etc.
- The model uses the new feature to smooth *average* prediction quality,
  improving MAE on typical days. But average-day errors don't cost
  revenue. Tail-day errors cost revenue.
- The forecast-gate's skip decision depends on predicted intraday
  spread. A slightly "flatter" forecast (lower variance) → more days
  pass the gate → more days the model commits to an incorrect timing
  choice on days when actuals surprise.
- Temperature and wind *actuals* (as they happened) would be more
  informative than the D-ahead forecast for MAE, but both are non-
  causal for dispatch timing (you can't use them at decision time).

**What doesn't work vs. what might:**

The current HRRR scope (1 cycle, 1 fxx, daily aggregate) is too coarse.
What would plausibly help:
- **Multiple fxx values per day** (F+3, F+9, F+15, F+21) so the model
  has explicit intraday shape information — e.g. "tomorrow midnight is
  warmer than dawn".
- **Forecast error features** — deviation of today's forecast from same
  month's average. Captures "unusual heat" signal directly.
- **Wind ramp features** — rate of change in forecast wind speed. This
  is a well-known scarcity predictor in ERCOT.
- **HRRR as input to a dedicated scarcity classifier**, with its
  probability then fed as a daily feature (instead of raw temps).

**What broke / what surprised me:**
- Initial result was confounded by too-short ffill (1h limit → HRRR
  NaN for most intervals). Fixed to 24h — result barely changed, which
  is the real story: HRRR as a daily-level feature just doesn't help
  intraday dispatch.
- RMSE improvement was tiny (0.1%). Expected either a larger MAE/RMSE
  win or a revenue win. Got a small MAE win and a large revenue loss.

**Decision for next step:**
Do not widen HRRR backfill until we've demonstrated HRRR *can* add
value on a different feature design. Next experiment: ERCOT Public
API's vintaged wind + solar forecasts (NP4-732-CD, NP4-737-CD).
These are ERCOT's OWN operational forecasts (STWPF, STPPF), not
external weather data, and are more directly tied to capacity margin
than air temperature. They should carry more dispatch-relevant signal.

**The session-best ML result stays at 56.4% of ceiling** (EIA + load +
prices + q50 + forecast-gate, 2019+ truncated training).

---

### 2026-04-24 — Truncated training wins: full-history vs 2019+ head-to-head

**Config:** q50 LightGBM, 200 iters, forecast-gate dispatch, same val
window (2020-11-01 → 2022-12-31, 791 days), same 27 features (prices
+ load + EIA-930 demand forecast + wind/solar lags). Walk-forward
monthly. `allow_nan_features=True` so LightGBM can handle pre-2019
EIA NaN rows.

**The question:** more training data (2011+) or feature-complete
training data (2019+)?

**Results on validation:**

| Variant | Training | Rows | Revenue   | % ceiling | MAE    | RMSE   | Wall clock |
|---------|----------|------|-----------|-----------|--------|--------|------------|
| A       | 2011-01+ (NaN pre-2019 EIA) | ~334k | \$9.75M | 50.6% | \$61.23 | \$626.07 | 226s |
| **B**   | **2019-01+** (truncated)    | **~120k** | **\$10.86M** | **56.4%** | **\$57.71** | **\$611.09** | **102s** |

**B wins on every metric.** Despite 3× less training data, the
truncated model:
- Earns +\$1.11M more on val (+5.8 percentage points of ceiling)
- Has lower MAE (\$57.71 vs \$61.23)
- Has lower RMSE (\$611 vs \$626)
- Finishes in half the wall-clock time

**This is the best ML result of the session.** 56.4% of ceiling beats
the prior best of 53.3% (q50 + gate on price+load only). Combined lift
of +3 pp from adding EIA features + truncating training compounds
cleanly.

**Interpretation — the non-stationarity thesis holds:**

Pre-2019 ERCOT is a different market: less wind/solar capacity,
different thermal generator mix, no winterization standards, a
mostly-offline battery fleet. LightGBM trained on that history learns
patterns that are actively misleading for 2020+ dispatch. The extra
2011-2018 rows are not "more data for the same problem" — they are
data from a meaningfully different conditional distribution, and
training on them drags the model toward behaviors that don't hold in
the val period.

This is specifically a problem for gradient-boosting on tabular data.
GBMs don't have any natural mechanism for down-weighting old regimes.
Every training row contributes equally to split choices.

**What this changes in the plan:**

1. **New default: train on 2019-01-01+ for any model using EIA or
   post-2019 features.** Document in DECISIONS.md.
2. **The original splits stay valid for price+load-only experiments**
   (they covered full 14-year history) — those numbers are still
   legitimate comparisons for the baselines.
3. **Future HRRR addition**: HRRR archive starts 2014-08. With
   truncated training at 2019+, HRRR data is always present — no
   NaN-handling needed, no distribution mismatch. Clean path.

**What broke / what surprised me:**
- Expected the NaN-fill variant to be within a couple pp of the
  truncated one. The gap (5.8 pp) is bigger than I'd have guessed,
  and the direction is unambiguous (B wins on MAE *and* RMSE *and*
  revenue — no interpretive trade-off).
- EIA wind/solar lag features ended up at similar importance to
  load_lag_1d in the truncated fit. Not game-changing but useful.

**Next:** update splits config + DECISIONS.md. Start HRRR narrow
backfill (1 cycle × 1 fxx, val window only, 8-way parallel, ~26 min)
to see if HRRR temperature forecasts compound on top of EIA features.

---

### 2026-04-24 — Data-source infrastructure landed

**Config:** no modeling run — this entry records the addition of four
new data-source fetchers. Everything still on validation set; test set
untouched.

**What:** wired up fetchers for the data sources identified in
DATA_GAP.md:

- FRED (Henry Hub gas price, no API key) — 7,353 daily obs 1997–present.
- EIA-930 region-data (hourly demand, day-ahead demand forecast, net
  generation, interchange for every US BA) — key in .env, 2024 smoke
  returns 35,183 rows with D/DF/NG/TI types.
- EIA-930 fuel-type-data (coal, gas, nuclear, wind, solar, hydro, other
  hourly) — works on same key. 2023 smoke: wind avg 12,331 MW, solar
  avg 3,638 MW, full year per fuel.
- NOAA HRRR weather forecasts via `herbie` + anonymous S3 — no key.
  Cycle 2024-06-15 12Z F03 Texas mean temperature = 81.7 F (reasonable).
- ERCOT Public API client (NP4-732-CD wind, NP4-737-CD solar,
  NP3-560-CD load forecast, NP3-233-CD outages) — scaffolded with ROPC
  auth. Blocked on ERCOT account username+password.

**Why this matters:** each of the DATA_GAP.md items identified as "most
likely to close the floor-to-model gap" is now a one-function-call away.
The next modeling session can immediately try: (1) EIA-930 day-ahead
demand forecast as a replacement for our current lagged-actual load
features; (2) HRRR Texas temperature forecast as an exogenous scarcity
driver; (3) FRED gas prices as a marginal-cost feature.

**What broke / what surprised me:**
- EIA-930 API 502s on large offset (>10k rows) — switched to
  month-by-month chunking + retry-with-backoff, which is fine.
- HRRR longitudes are in 0..360° on S3 not -180..180 — one-line fix.
- Herbie bundles eccodes in its wheel; no `brew install eccodes` needed.

**Next modeling step (clear):** refit q50 walk-forward with the real
EIA-930 day-ahead demand forecast and HRRR temperature forecasts as
features, keeping the forecast-gate dispatch. First honest test of
whether exogenous vintaged forecasts actually beat the floor. Timing:
~4–6 hours of integration + walk-forward compute.

---

### 2026-04-24 — Combined strategy (q50 + load + scarcity-prob feature + gate)

**Config:** two-stage walk-forward.
- Stage 1: walk-forward LGBM scarcity classifier over the full history
  (monthly refits, 4,382 days), producing one out-of-sample scarcity
  probability per local date.
- Stage 2: walk-forward q50 LightGBM point forecaster with 21 features
  — 15 from prices + calendar, 5 from load, 1 is `scarcity_prob_today`
  (broadcast from the daily stage-1 probs).
- Dispatch: forecast-gate (skip days whose forecast simulates to ≤ 0).

Script: `scripts/run_combined.py`. Validation window: 791 days.

**Results (val):**

| Strategy                              | Revenue    | % ceiling | Sharpe | Worst day   |
|---------------------------------------|------------|-----------|--------|-------------|
| ceiling                               | \$19.27M   | 100.0%    | 0.27   | \$0         |
| floor                                 | \$16.77M   | 87.0%     | 0.21   | −\$830,710  |
| q50 + gate (previous best, no scarcity feature) | \$10.28M | **53.3%** | 0.34 | −\$8,991 |
| **combined: q50 + load + scarcity-prob feature + gate** | \$10.12M | 52.5% | 0.34 | −\$9,296 |
| combined: q50 + load + scarcity-prob feature (no gate) | \$7.68M | 39.8% | 0.11 | −\$1.39M |

Point forecast quality of the combined model:
- MAE: **\$63.52** (worse than \$58.53 without scarcity_prob feature)
- RMSE: \$637.93 (worse than \$619.30)

**What this negative stacking result tells us:**

1. **Stacking didn't stack.** Adding the scarcity probability as a
   per-day broadcast feature slightly degraded both point forecast
   accuracy *and* dispatch revenue. Going from 53.3% → 52.5% of
   ceiling on gated is inside the noise, but the fact that it didn't
   *improve* is the finding.

2. **Why plausibly:** the scarcity probability is (a) noisy, particularly
   in early years when the classifier's training set is small, and (b)
   constant across all 96 intervals within a day, which is an awkward
   signal for a 15-min forecaster. The model does not gain intraday
   timing information from a constant daily feature, only level info
   — and that level signal is already present in the lag/rolling price
   features.

3. **The session-best technique remains the forecast-gate alone.** The
   dispatch layer is doing most of the interesting work:
   - It prevents tail losses (−\$9k vs −\$1.36M worst day, 100× better).
   - It roughly doubles Sharpe (0.34 vs 0.18).
   - And it does this with *any* reasonable point forecaster — L1 LGBM,
     q50 LGBM, with or without load, with or without scarcity feature.

4. **The natural-spread floor is a stubborn ceiling for all ML
   variants tested.** Six distinct ML variants (prices-only, prices+load,
   scarcity-rule, scarcity-oracle-rule, q50 bare, q50+gate, combined)
   range from 23.6% to 53.3% of the perfect-foresight ceiling, all below
   the 87% captured by a mechanical oracle that cheats by knowing
   realized prices. **No ML variant beat the floor on validation.**

5. **The remaining gap is structural, not algorithmic.** With only
   historical prices + lagged load, the forecaster cannot distinguish
   a normal summer day from a day where a thermal unit will trip at
   14:32 and prices will spike to \$5,000 by 15:00. Closing the gap
   requires real-time exogenous information — short-term load forecasts
   with hourly vintage, wind/solar production forecasts, system
   operating reserves, temperature forecasts — which aren't available
   as historical archives via gridstatus.

**Overall session honest takeaway:**

- The *mechanical* natural-spread floor at 87% of ceiling is a very
  strong baseline for ERCOT BESS arbitrage. It is not trivial to beat
  with price-only ML.
- A forecast-gate that skips days the model doesn't like is the one
  technique that reliably moved our forecast-driven numbers by enough
  to matter. Worth keeping.
- Revenue is dominated by scarcity days; scarcity prediction with the
  features we can retrieve improves over random but is not strong
  enough to drive dispatch lift.
- The realistic next step is feature acquisition (real forecast
  archives for load / wind / solar), not more modeling on the same
  features. Until then, every further model on price-history-only will
  produce more variants of the 40-55% of ceiling band we've mapped out.

**Test set status:** untouched. With what we have now, there is no
reason to touch it — no strategy beats the floor on validation, so
"best on val" is unpromising enough that a fresh holdout number would
not change the conclusions.

---

### 2026-04-24 — Quantile LightGBM + forecast-gated dispatch

**Config:** three LightGBM quantile fits (tau = 0.1, 0.5, 0.9), 200 iters
each, walk-forward monthly. Features: 20 cols (prices + load lags +
calendar). Script: `scripts/run_quantile.py`.

**Data:** HB_NORTH validation window (791 days).

**Quantile forecast metrics:**
- q10 pinball loss: 7.00
- q50 pinball loss: 29.32 (same ballpark as L1 MAE \$58)
- q90 pinball loss: 37.71
- **80% prediction-interval coverage: 77.0%** (vs target 80.0%)
- Mean q90−q10 width: \$51.75

Coverage is close to nominal; the model is mildly over-confident on
interval width but substantially accurate. The interval is useful.

**Dispatch comparison (val):**

| Rank | Strategy                        | Revenue    | % ceiling | Sharpe | Worst day   | Lift vs floor |
|------|---------------------------------|------------|-----------|--------|-------------|---------------|
| 1    | perfect_foresight_ceiling       | \$19.27M   | 100.0%    | 0.27   | \$0         | +\$2.51M      |
| 2    | natural_spread_floor            | \$16.77M   | 87.0%     | 0.21   | −\$830,710  | 0             |
| 3    | **q50 + forecast-gate**         | \$10.28M   | **53.3%** | **0.34** | **−\$8,991** | −\$6.49M    |
| 4    | q50 alone (threshold dispatch)  | \$7.99M    | 41.5%     | 0.12   | −\$1.36M    | −\$8.77M      |
| 5    | q_spread_gate (explicit q90-q10)| \$7.99M    | 41.5%     | 0.12   | −\$1.36M    | −\$8.77M      |

**The key finding — dispatch gating matters more than quantile width:**

1. **q50 + forecast-gate is the best forecast-driven result of the
   session.** 53.3% of ceiling vs. 47.6% (L1 LGBM + load) and 41.6%
   (L1 LGBM prices only). The technique: let the forecaster simulate
   its own day's schedule against its own predictions; if the predicted
   net is ≤ 0, skip that day entirely. No quantile knowledge required
   for this lift — the point forecast alone is enough to gate.

2. **Sharpe doubles and the worst day improves 100×.** 0.34 vs 0.18
   (LGBM+load), worst day −\$9k vs −\$1.36M. For any operator that
   reports risk-adjusted metrics or drawdown-sensitive covenants, this
   is a bigger win than the headline revenue number suggests.

3. **The explicit q90−q10 gate didn't trigger once.** Breakeven spread
   I computed (~\$9/MWh) is well below every day's predicted spread
   because the q90 of afternoon prices is always non-trivially above
   the q10 of morning prices. To make quantile width actionable, we
   need a tighter gate — e.g. compare predicted net revenue intervals
   rather than raw price spreads, or require the *lower bound* of
   predicted revenue to clear costs.

4. **Quantile forecasts have latent value we didn't exploit here.**
   Good 80% coverage means q10/q90 are informative about uncertainty.
   A proper exploitation — CVaR-adjusted dispatch, conformalized
   prediction bands used to size daily cycle — would likely extract
   more lift. Explicit TODO.

**What broke / what surprised me:**
- Forecast-gated q50 dispatches on only a subset of days and yet earns
  nearly as much as q50 on all days. The days it skips are systematically
  loss-making — the point forecaster *can* tell when it's unconfident,
  it just doesn't have to be told via quantiles.
- Per-day LGBM quantile fits are ~3 min each on val (200 iters × 27
  refits × 3 quantiles = 10 min total). Acceptable; would need to be
  amortized if scaling to test or full-history.

**Revised ranking of what's been tried on val (best → worst revenue):**

| Strategy                             | Revenue    | % ceiling | Lift vs floor |
|--------------------------------------|------------|-----------|---------------|
| perfect_foresight_ceiling            | \$19.27M   | 100.0%    | +\$2.51M      |
| natural_spread_floor                 | \$16.77M   | 87.0%     | 0             |
| q50 + forecast-gate                  | \$10.28M   | 53.3%     | −\$6.49M      |
| LGBM + load features                 | \$9.17M    | 47.6%     | −\$7.60M      |
| Scarcity classifier + rule (worse)   | \$8.65M    | 44.9%     | −\$8.11M      |
| LGBM prices-only                     | \$8.03M    | 41.6%     | −\$8.74M      |
| q50 only (no gate)                   | \$7.99M    | 41.5%     | −\$8.77M      |
| Scarcity oracle + rule               | \$4.55M    | 23.6%     | −\$12.22M     |

**Next:**
1. Combine the two best ideas: LGBM + load features + forecast-gate +
   scarcity-probability as a FEATURE (not a switch). Expect further lift.
2. Proper quantile-based CVaR dispatch.
3. Exogenous wind/solar — would need EIA-930 or a forecast archive.
4. Only after the above beats the floor convincingly: touch the test set.

---

### 2026-04-24 — Scarcity-day classifier + rule-based dispatch (honest negative result)

**Config:** binary LightGBM (is_unbalance=True, AP metric) trained
walk-forward monthly on daily features (lagged price aggregates + daily
load peaks + calendar + days-since-scarcity). Scarcity defined as
`daily max RTM SPP > \$500`. Dispatch integration: on days with
p_scarcity > 0.5, replace LGBM forecast-driven schedule with a fixed
"hold for peak" rule — charge 01:00-07:00 local, discharge 15:00-21:00
local, both at full rated power.

**Data:** HB_NORTH validation window (791 days). Overall scarcity rate
on val = 11.5% (91/791 days).

**Classifier quality:**
- **PR-AUC: 0.220** (baseline = 0.115, so ~2× random)
- Precision @ recall=0.3: 0.269
- Precision @ recall=0.5: 0.145
- At threshold 0.5: TP=18, FP=50, FN=73, TN=650 (recall 19.8%,
  precision 26.5%)

The classifier learns real signal from price + load history — it is
meaningfully better than flipping a weighted coin. But it is not good
enough to replace a LightGBM fit in point-forecast mode.

**Dispatch comparison (val):**

| Rank | Strategy                                       | Revenue    | % ceiling | Lift vs floor |
|------|------------------------------------------------|------------|-----------|---------------|
| 1    | ceiling                                        | \$19.27M   | 100.0%    | +\$2.51M      |
| 2    | floor                                          | \$16.77M   | 87.0%     | 0             |
| 3    | LGBM + load (no classifier)                    | \$9.17M    | 47.6%     | −\$7.60M      |
| 4    | **Scarcity-aware (classifier + rule)**         | \$8.65M    | 44.9%     | −\$8.11M      |
| 5    | **Scarcity oracle (perfect classifier + rule)**| \$4.55M    | 23.6%     | −\$12.22M     |

**What this negative result tells us:**

1. **The rule-based "hold then dump" override is strictly worse than
   the LGBM schedule** — even with a PERFECT (oracle) scarcity classifier.
   23.6% of ceiling < 47.6% of ceiling. A good classifier paired with a
   bad dispatch rule destroys value.

2. **The fixed 01:00-07:00 / 15:00-21:00 window mis-times many scarcity
   events.** ERCOT scarcity doesn't peak at a fixed clock hour — it
   peaks when net load (load minus renewables) hits the capacity ceiling,
   which shifts by season and weather pattern. The fixed-window rule
   often places discharge at the wrong hours even on true scarcity days.

3. **Rigid rule also overtrades.** The rule tries to charge for 6h and
   discharge for 6h at full power — way beyond the 200 MWh usable. The
   battery hits its SOC cap very quickly, then most of the rule is
   clipped. Clipped-interval rate doubled (6-7% vs 3% for LGBM), all
   wasted cycles.

4. **The right way to use a scarcity signal is as a FEATURE, not a
   SWITCH.** A future experiment: include `p_scarcity_today` as an
   input feature to the per-interval LGBM forecaster. The forecaster
   then learns by itself how to inflate afternoon prices on likely-
   scarcity days. Tested separately from this report.

5. **This is exactly the pattern the session has been uncovering.** A
   naive-looking improvement (add a classifier, override bad days)
   destroys value when it doesn't respect the battery's physics and
   the market's irregular peak timing. The floor baseline already
   reflects *ideal timing under a cycle cap*; any rigid rule fights
   that rather than extending it.

**What broke / what surprised me:**
- Expected scarcity_oracle to be an UPPER bound on classifier-aware
  dispatch. It's a LOWER bound of that approach instead, because the
  dispatch *rule* itself is the problem.
- My first run crashed on an index length mismatch because the
  classifier drops warmup rows. Fixed by aligning the eval mask to the
  classifier's (reduced) index rather than the daily features index.

**Next:** skip the rule-override approach entirely. Move to probabilistic
(quantile) point forecasts and a *spread-uncertainty* dispatch gate that
scales cycle size with forecast confidence.

---

### 2026-04-24 — LightGBM + ERCOT load features on validation

**Config:** same battery (100 MW / 200 MWh), same 30-day walk-forward,
same L1 LGBM. Added 5 exogenous features from hourly ERCOT system-wide
load (actuals only — no historical forecasts available from gridstatus):
`load_lag_1h`, `load_lag_1d`, `load_lag_1w`, `load_roll_mean_1d_left`,
`load_rel_to_7d_mean`. Total feature count: 15 → 20.

**Data:** HB_NORTH RTM SPP val window (791 days). Plus hourly ERCOT load
2011–2022 cached via `get_hourly_load_post_settlements`.

**What:** rerun of the previous LGBM walk-forward with load features
added. Load is forward-filled from hourly to 15-min granularity. All load
features are lagged (≥ 1h) so no same-time leakage.

**Results (val window):**

| Strategy                  | Revenue    | % ceiling | Lift vs floor | Δ vs LGBM no-load |
|---------------------------|------------|-----------|---------------|-------------------|
| perfect_foresight_ceiling | \$19.27M   | 100.0%    | +\$2.51M      | —                 |
| natural_spread_floor      | \$16.77M   | 87.0%     | 0             | —                 |
| lgbm + load features      | \$9.17M    | **47.6%** | −\$7.60M      | **+\$1.14M**      |
| lgbm (prices only)        | \$8.03M    | 41.6%     | −\$8.74M      | baseline          |

Forecast quality: MAE \$58.53 (was \$57.68), RMSE \$619.30 (was \$605.42).

**What this tells us:**

1. **Exogenous features help dispatch without helping point-forecast
   accuracy.** MAE moved up slightly; RMSE moved up slightly. But
   dispatch revenue improved \$1.14M (+6 pp of ceiling). The model uses
   load features to pick the right intervals to charge/discharge, even
   when it's no better at predicting the level.

2. **We're still \$7.6M below the floor.** The floor captures 87% of
   ceiling; our best ML now captures 47.6%. Load-only exogenous signals
   don't close the gap. Scarcity prediction — which depends on capacity
   margin (load + outages − renewables) — is the missing piece, and
   ERCOT doesn't archive historical wind/solar at the granularity we
   need via gridstatus.

3. **Tail losses reduced.** Worst day: −\$1.36M → −\$1.18M. Better
   feature information made the model less over-confident on bad days.

4. **Sharpe improved from 0.12 to 0.18.** The distribution of daily
   revenue is tighter with load features — the model makes smaller, more
   consistent bets.

**What broke / what surprised me:**
- `gridstatus.Ercot.get_wind_actual_and_forecast_hourly` and its solar
  analog return "no documents" for any historical date — the MIS
  archives for these roll off, only ~8 days retained. Historical load
  works via a separate `get_hourly_load_post_settlements` path pointing
  to a different ERCOT archive. Documented in DATA.md as an open
  question for phase 2: EIA-930 or self-hosted forecast snapshotting
  would be the next move if we pursue wind/solar features.
- Improvement on MAE was ~zero despite adding informative features.
  This is a useful calibration: when evaluating future models, do not
  rely on MAE alone — score everything on revenue lift.

**Next:** scarcity day classifier (binary), then quantile forecasts.

---

### 2026-04-24 — LightGBM walk-forward on validation window

**Config:** default battery (100 MW / 200 MWh). Split per
`configs/splits.yaml` (train 2011-01-01→2020-10-31, val
2020-11-01→2022-12-31). Monthly refit. LightGBM L1 objective, 500 iters,
default hyperparameters (no tuning yet). Features: 7 lag columns (15min,
1h, 4h, 1d, 2d, 1w, 2w), 3 rolling stats (1d mean+std, 1w mean), 5
calendar features. Script: `scripts/run_lgbm_walkforward.py`.

**Data:** HB_NORTH RTM SPP, validation window = 791 days, 75,940 intervals.

**What:** first fitted ML model evaluated under proper walk-forward
discipline. Monthly retrains; predictions at time *t* use only data
from strictly before the most recent retrain boundary. Forecasts feed
into the same threshold-rule dispatch (charge in predicted-cheapest
intervals, discharge in predicted-most-expensive).

**Results on validation window:**

| Strategy               | Revenue       | % ceiling | Lift vs floor |
|------------------------|---------------|-----------|----------------|
| perfect_foresight (val)| \$19.27M      | 100.0%    | +\$2.51M       |
| natural_spread_floor   | \$16.77M      | **87.0%** | 0              |
| lgbm_walkforward       | \$8.03M       | 41.6%     | **−\$8.74M**   |

**Forecast quality (val):** MAE \$57.68, RMSE \$605.42, n = 75,940.

**What this confirms:**

1. **LightGBM with lag+calendar features does not beat the floor.**
   It captures 41.6% of ceiling — about the same as persistence (39.7%)
   and seasonal-naive (42.5%) on the full history. Refitting monthly did
   not change the fundamental picture: without information beyond
   historical prices, forecast-driven dispatch destroys value compared
   to the mechanical oracle.

2. **Point-forecast accuracy is not the right target.** MAE of ~\$58 on a
   mean-\$41 market sounds decent. But RMSE is \$605 — errors are heavy-
   tailed, and the big misses are *precisely* on scarcity days where
   revenue lives. A model optimizing MAE learns the low-volatility
   central tendency well and misses the tail entirely.

3. **The ML path forward is not "bigger model" — it's "different
   features."** Price history alone cannot predict the events that
   drive 66% of revenue. To beat the floor, the forecaster needs
   exogenous information that actually encodes tail-risk: forecasted
   load vs capacity margin, wind/solar forecast error distributions,
   temperature extremes, outage reports, real-time operating reserves.
   Without these, additional modeling effort on the same features will
   return diminishing noise.

**What broke / what surprised me:**
- RMSE (\$605) being 10× MAE (\$58) is the smoking gun for tail-error
  insensitivity under L1. Switching to L2 (squared error) might make
  the number look better but wouldn't fix the fundamental feature-
  coverage gap. Documented but not pursued.
- Walk-forward took 382s for 791 days of 30-day refits (~27 retrains).
  LightGBM fitting is fast enough that this is not a blocker for
  iteration, but probabilistic methods (quantile regression per
  forecast) would need reconsideration.
- LGBM's worst day was −\$1,354,759 — worse than persistence's
  −\$1,757,514 by a hair, but of the same order. Bad forecasts on
  tail-event days turn into catastrophic dispatch errors.

**Next (revised priority):**
1. **Exogenous features.** Add ERCOT load forecasts (STLF), renewable
   forecasts (STWPF / STPPF), and basic weather (ERA5 temperature
   anomalies by zone). This is the material lever on scarcity prediction.
2. **Scarcity classification as a separate problem.** Train a binary/
   multiclass model for "will there be a spike today?" and fold the
   probability into dispatch (stay full going in, not cycling before).
3. Probabilistic forecasts (quantile regression / LGBMLSS / conformal)
   so dispatch can act on uncertainty.
4. Hyperparameter tuning on validation only, after the new features
   are in — tuning before exogenous features would just overfit the
   central-tendency task that doesn't matter for revenue.

**Test set status:** untouched. Per METHODOLOGY §1.

---

### 2026-04-24 — forecast-driven baselines and gated floor, HB_NORTH 2011–2024

**Config:** default battery (100 MW / 200 MWh / η=0.85 / \$2/MWh deg),
1 cycle/day, `tz=US/Central`. Script: `scripts/run_baselines_real.py`.

**Data:** same — HB_NORTH RTM SPP, 2011–2024, 490,943 intervals, 5,114 days.

**What:** added three strategies and ran the full comparison against the
ceiling:
- **Gated natural-spread** — same as floor but skips days whose simulated
  net revenue (from initial SOC state, using realized prices) is ≤ 0.
- **Persistence forecast** — Forecast(t) = Price(t − 1 day); applied to
  the threshold-rule dispatch.
- **Seasonal-naive forecast** — Forecast(t) = median of Price(t − 7·k days)
  for k = 1..4; applied to the threshold-rule dispatch.

**How:** all five strategies use the same 1-cycle-per-day budget and the
same execution simulator. Forecast-driven strategies build the schedule
from forecast, then execute on realized prices. No fitting, no train/test
split needed — all forecasts are strictly backward-looking.

**Headline: % of ceiling captured:**

| Rank | Strategy                      | Revenue    | % ceiling | Lift over floor |
|------|-------------------------------|------------|-----------|-----------------|
| 1    | perfect_foresight_ceiling     | \$81.31M   | 100.0%    | +\$8.10M        |
| 2    | natural_spread_gated (oracle) | \$73.58M   | 90.5%     | +\$0.37M        |
| 3    | natural_spread_floor          | \$73.22M   | 90.0%     | 0               |
| 4    | seasonal_naive_4w             | \$34.56M   | 42.5%     | **−\$38.65M**   |
| 5    | persistence                   | \$32.26M   | 39.7%     | **−\$40.95M**   |

**Key findings:**

1. **Forecast-driven strategies with naive forecasts LOSE money vs the
   floor.** Persistence does −\$40.95M over 14 years; seasonal-naive
   −\$38.65M. Capturing yesterday's shape and betting on it is a net
   value destroyer in ERCOT RTM. The reason: ERCOT's intraday price
   shape shifts a lot day-to-day — heat events, wind ramps, and
   scarcity patterns don't repeat on a 24h clock.

2. **The gate is nearly free money but the size is small.** Gated floor
   improves over floor by only \$365k over 14 years (+0.5% of ceiling).
   Most bad days in ERCOT are *scarcity-or-negative* days where cycling
   is still worth it; the gate only helps on truly flat, low-price days
   which are rare in ERCOT.

3. **Forecast-driven worst-day: −\$1.76M.** Compare to gated floor's
   −\$316k. A bad forecast aimed at the wrong intervals on a volatile
   day compounds into a large directional loss. Any ML model must be
   evaluated for *tail behavior*, not just mean performance.

4. **Regime concentration holds across strategies.** Scarcity days
   supply 66-72% of revenue for every strategy. Forecast-driven
   strategies earn a slightly higher *share* from scarcity (~72%)
   precisely because their normal-day performance is worse — the
   denominator is smaller, not the numerator bigger. They capture
   ~44% of the *dollar amount* of ceiling scarcity revenue, vs floor's
   91%.

5. **The ML bar is clear.** An ML model earns its place only if its
   forecast-driven dispatch revenue exceeds the natural-spread floor.
   Beating persistence is trivial; beating the floor is the real test.
   This is a much harder bar than "low MAE" — a model can have
   respectable MAE and still lose to the floor because its errors are
   correlated with intraday timing.

**What broke / what surprised me:**
- Persistence and seasonal-naive perform similarly (~40-43%), despite
  the latter using 4x more history. The "use same DOW" assumption
  doesn't add much over "use yesterday" in ERCOT's noisy environment.
- Scarcity days: floor captures \$48.3M on them, forecast-driven only
  \$23.2M (persistence) / \$23.9M (seasonal-naive). Naive forecasters
  miss scarcity events half the time. The alpha opportunity for ML is
  specifically *predicting tail events*, which is exactly what our
  ML models will need to focus on.

**Next:**
1. Walk-forward evaluation harness (so we can fit models on train data,
   forecast on hold-out, roll monthly).
2. First fitted baseline: LightGBM with lag + calendar features, scored
   against the floor — if it can't beat the natural-spread floor on
   validation, we learn something important early.
3. Probabilistic forecasts + a skill-weighted dispatch gate that uses
   forecast uncertainty to decide when to cycle.

---

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
