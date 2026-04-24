# Battery Arbitrage on ERCOT / CAISO — Project Plan

## 1. Goal
Build an ML system that forecasts short-horizon electricity prices (LMPs) on a US
ISO (ERCOT or CAISO) and produces a charge/discharge schedule for a simulated
grid-scale battery (BESS), evaluated against realistic operational baselines.

## 2. Why it's promising
- Real, multi-billion-dollar industry (Tesla Megapack, Fluence, GridStor, Tyler Tech).
- Forecast quality translates directly into revenue — economics are unambiguous.
- Public, free, numerical data going back 15+ years; no news, scraping, or text.
- Published ML work on BESS dispatch is thinner than equities/crypto, so a careful
  solo researcher can produce novel results.
- Dual path: publishable research and/or commercially deployable strategy.

## 3. Guiding principle: keep phase 1 tiny
The biggest risk is over-engineering before we understand how much revenue comes
from the *existence of price spreads* versus any forecast we build. Phase 1 must
be minimal and finishable in days, not weeks:

- One ISO, one node, RTM only.
- Point forecast, not distribution.
- Threshold rule for dispatch (charge below X, discharge above Y), not LP.
- Full evaluation protocol in place from day one.

Only after phase 1 is honestly measured against baselines do we add complexity.

## 4. Problem definition
Two coupled sub-problems — keep them separate so we always know whether gains
come from better forecasts or better dispatch logic.

**4a. Price forecasting.** Given information available at time *t*, forecast
5-minute RTM LMPs at a chosen node over the next *H* intervals.

**4b. Dispatch optimization.** Given a forecast + battery physics (power rating,
energy capacity, round-trip efficiency, SOC limits, cycle/degradation cost),
choose a charge/discharge schedule.

Scope decisions to fix up-front:
- Market: ERCOT (leaning) or CAISO.
- Node: one representative settlement point.
- Battery spec: e.g. 100 MW / 200 MWh, 85% round-trip, 1-cycle/day soft limit.
- Market participation: RTM only in phase 1. DAM + RTM and ancillary services later.

## 5. Data
- ERCOT: SCED/RTM LMPs, DAM LMPs, load, wind/solar generation, outages. Public reports.
- CAISO: OASIS (LMPs, load, renewables). Public.
- Weather: ERA5 reanalysis historically, **plus the ISO's own published forecasts**
  for realistic feature availability.
- Calendar features (hour, DOW, month, holidays).

## 6. Baselines (ship these before any ML)

Any ML model must be measured against *all* of these, not just a naive one.

1. **Natural-spread baseline.** Charge at the lowest-price hour of each day,
   discharge at the highest. **No forecast.** Pure oracle over daily min/max on
   *realized* prices. This tells us how much revenue is sitting in the raw price
   spread itself, independent of prediction skill. Critical: if our ML model
   doesn't meaningfully exceed this, the model is adding nothing.
2. **Persistence.** Price at *t+h* = price at *t*, or same interval previous day.
3. **Seasonal naive.** Same hour, same DOW, trailing N weeks median.
4. **Classical forecast + threshold rule.** SARIMA or ExpSmoothing → charge below
   the forecast's 20th percentile hour, discharge above the 80th.
5. **Published benchmark.** At least one result from recent literature on the
   same ISO for sanity-checking our forecasting metrics.

### Not a baseline — a ceiling
**Perfect-foresight dispatch.** Optimal schedule given *true* future prices. It
will always win. Use it only to measure headroom: "our strategy captures X% of
the theoretical maximum." Do not list it alongside baselines we're trying to beat.

## 7. Modeling approach
Strict progression. Do not skip ahead until the previous phase has been measured.

1. Classical point forecast (SARIMA, ExpSmoothing, LightGBM with lag + calendar +
   exogenous features) → threshold dispatch rule.
2. Same forecasts → LP-based dispatch (cvxpy) with battery constraints. Compare
   against threshold rule to isolate optimizer contribution.
3. Probabilistic forecasts (quantile regression, conformal prediction) → risk-aware
   dispatch using the forecast distribution.
4. Deep models (temporal fusion transformer, N-BEATS, DeepAR) on the same features.
5. (Stretch) Decision-focused / end-to-end learning where the forecaster is
   trained to minimize dispatch regret directly, not forecast error.

## 8. Evaluation protocol
- **Splits, chronological only.**
  - Train: earliest ~70% of history.
  - Validation: next ~15%, used for model selection / hyperparameter tuning.
  - Test: most recent ~15%, touched **once** at the very end.
- **Walk-forward evaluation** on the test set. Required. Retrain at a cadence that
  matches reality (e.g. monthly retrain, daily forecast, 5-min dispatch). Report
  mean and dispersion of metrics across windows, not just a single number.
- **Forecasting metrics.** MAE, RMSE, pinball loss (once probabilistic), CRPS,
  directional accuracy.
- **Dispatch metrics.**
  - $/MWh/day revenue.
  - % of perfect-foresight captured (headroom).
  - Sharpe across days; worst-day drawdown.
  - **Missed opportunity cost** — $ left on the table from not charging when cheap
    or not discharging when expensive, broken out separately from losses due to
    bad actions.
- **Report by regime.** Normal, scarcity, negative-price days. Averages hide tails.

### Revenue attribution (mandatory)
Decompose every strategy's revenue into:
- What you would have earned from the **natural daily spread** (baseline #1) —
  i.e. the "free" revenue sitting in the market.
- What came from **forecast skill** — the lift over baseline #1 attributable to
  predicting *which* hours to charge/discharge.
- What came from **optimization** — the lift from LP/constraint-aware dispatch
  over a simple threshold rule, holding the forecast fixed.

If the "forecast skill" bucket is small, the ML is not earning its keep and we
should say so out loud.

## 9. ML pitfalls we must not make
Reread before every experiment.

- **Look-ahead leakage.** A feature at time *t* must use only data timestamped
  ≤ *t − publication delay*. ERCOT SCED prices are published with a lag; respect it.
- **Publication delays and revisions.** ISO data is frequently revised post-hoc.
  Explicitly log, per feature: *what value was known at time t*, not what the
  final archived value says today. Prefer "as-of" snapshots when available.
- **Forecast vs. actual leakage.** If a weather or load feature exists as both a
  forecast (available pre-decision) and an actual (only post-decision), **use the
  forecast** in training. Audit every feature against this rule.
- **Own-impact ignored.** A battery's bids move prices. Small sizes on ERCOT are
  usually negligible, but state the assumption and test sensitivity.
- **No random K-fold.** Ever. Time series → chronological splits and walk-forward.
- **No shuffled batches across time boundaries** in deep models unless the
  windowing explicitly preserves temporal order per sample.
- **Standardization leakage.** Compute means/stds/encoders on the training window
  only, then apply forward. Refit per walk-forward step.
- **Target leakage via engineered features.** Rolling stats, calendar-aligned
  aggregates, "last known price" — audit each for whether it could include *t*.
- **Regime overfitting.** 2021 Texas winter storm and 2022–23 gas spikes are
  extreme. Don't drop them, but report metrics with and without extreme days and
  confirm the model isn't silently memorizing them.
- **Backtest sloppiness.** Always include: round-trip efficiency, SOC limits,
  cycle limits, degradation cost ($/MWh throughput), bid/offer rules, settlement
  timing. A "revenue" number without these is not a number.
- **Cherry-picked seed.** Report mean ± std over ≥ 5 seeds for stochastic models.
  Commit to the seed policy before seeing results.
- **Test-set peeking.** Test set is touched once. If you look, you've burned it —
  start a fresh holdout or wait for more data.
- **Metric gaming.** Optimizing MAE yields a different model than optimizing
  dispatch revenue. State the objective, then measure both.

## 10. Milestones
1. Data pipeline — one ISO, one node, clean RTM LMP + load + renewables.
2. Baseline #1 (natural spread) measured — establishes the revenue floor.
3. Baselines #2–#4 (persistence, seasonal, classical + threshold) measured under
   full protocol.
4. Classical ML forecast + threshold rule beats baselines on validation.
5. LP-based dispatch — measure optimizer lift over threshold rule.
6. Probabilistic forecast + risk-aware dispatch.
7. Deep model comparison.
8. (Stretch) Decision-focused / end-to-end learning.
9. Write-up, including honest discussion of what did and did not beat baselines,
   and full revenue attribution.

## 11. Open questions
- ERCOT or CAISO first? (Leaning ERCOT — more volatility, simpler market design.)
- Which node(s)? One hub to start, or a small basket?
- DAM + RTM jointly, or RTM-only in phase 1? (Leaning RTM-only.)
- Battery spec — fix at 100 MW / 200 MWh, or sweep?
- Ancillary services (RegUp/RegDown, responsive reserve) — in scope or later?
- Cycle/degradation model — fixed $/MWh or state-of-health-dependent?