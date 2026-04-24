# Methodology

Single source of truth for how we split data, run experiments, and score
results. Every experiment must conform to this document. If the methodology
changes, update this file *first* and note the change in [DECISIONS.md](DECISIONS.md).

## 1. Data splits

Chronological only. No random sampling. No K-fold.

- **Train:** earliest ~70% of the available history at the node.
- **Validation:** next ~15%. Used for model selection, feature choice, and
  hyperparameter tuning.
- **Test:** most recent ~15%. Touched **once**, at the very end of a modeling
  phase. If you peek, the test set is burned — start a new holdout or wait for
  more data to accumulate.

Exact cutoff dates for the active node are recorded in
`configs/splits.yaml` and must not be changed mid-project without a
[DECISIONS.md](DECISIONS.md) entry.

## 2. Walk-forward evaluation

Required on the test set. A single-shot train/test number is not acceptable.

- **Refit cadence:** monthly (configurable). Each walk-forward window refits
  the model on all data up to the window start, then forecasts and dispatches
  over the window.
- **Forecast cadence:** daily (produce a forecast at a fixed hour each day for
  the next 24h of 5-min intervals). This mimics how an operator would actually
  run the model.
- **Dispatch cadence:** 5-min, inside the forecast window.
- **Reporting:** mean, std, min, max of each metric across walk-forward
  windows. Single-number averages hide regime-dependent behavior.

## 3. Feature availability rule

A feature used at time *t* must depend only on information available at time
*t − d*, where *d* is the feature's **publication delay**. See [DATA.md](DATA.md)
for per-feature delays. Violations are the most common cause of leakage.

When a quantity exists in two forms — *forecast* (available pre-decision) and
*actual* (available only post-decision) — always use the forecast in training
features. The training-time actual is leakage.

## 4. Baselines

Every ML run must report metrics against all four reference strategies. See
[PLAN.md](PLAN.md) §6 for details.

1. Natural-spread (charge at realized daily min, discharge at daily max — no
   forecast). The floor.
2. Persistence.
3. Seasonal-naive.
4. Classical forecast (SARIMA / ExpSmoothing) + threshold dispatch rule.

Perfect-foresight is the ceiling, not a baseline — report `% of ceiling
captured`, don't try to beat it.

## 5. Metrics

### 5.1 Forecasting
- MAE, RMSE (point forecasts).
- Pinball loss, CRPS (once probabilistic forecasts exist).
- Directional accuracy (sign of hourly change).
- Report overall and **broken out by regime**: normal, scarcity (>$500/MWh
  days), negative-price periods.

### 5.2 Dispatch
- **$/MWh/day revenue** — primary metric.
- **% of perfect-foresight** — headroom indicator.
- **Sharpe across days** — revenue mean / revenue std over days.
- **Worst-day drawdown.**
- **Missed opportunity cost** — revenue left on the table from failing to
  charge in the cheapest hours or discharge in the most expensive, reported
  separately from losses from wrong actions.
- Report by regime, same as forecasting.

### 5.3 Revenue attribution (mandatory)
For every strategy, decompose daily revenue into:

- **Spread component:** revenue earned by the natural-spread baseline on the
  same day. The "free" money from the market.
- **Forecast-skill component:** (strategy revenue with threshold rule) −
  (spread baseline revenue), holding dispatch rule fixed.
- **Optimization component:** (strategy revenue with LP dispatch) − (strategy
  revenue with threshold rule), holding forecast fixed.

If the forecast-skill component is near zero or negative on average, the model
is not earning its keep. Report it regardless of how embarrassing it is.

## 6. Battery simulation

Every backtest must include, at minimum:

- Power rating (MW), energy capacity (MWh), starting SOC.
- Round-trip efficiency (η, typically 0.85–0.90). Applied on charge and discharge.
- SOC bounds (soc_min, soc_max), typically 5% / 95%.
- Cycle limit (soft, e.g. 1 full cycle/day) or degradation cost in $/MWh-throughput.
- Settlement timing — when revenue is realized, when SOC updates.

Parameters are specified in `configs/battery.yaml`. A "revenue" number produced
by a sim that omits any of the above is not a valid number.

## 7. Reproducibility rules

- **Seeds.** Every stochastic run logs its RNG seed. Reported numbers are
  mean ± std over ≥ 5 seeds. Commit to the seed policy before seeing results.
- **Config.** Every experiment is launched from a versioned config file
  (YAML in `configs/`). The config hash is logged with results.
- **Data snapshot.** Raw data files are pinned by filename + SHA256.
  Revised ISO data must not silently replace older snapshots. See [DATA.md](DATA.md).
- **Code version.** The git commit SHA is logged with each experiment.
- **Run manifest.** Every experiment writes a `results/<run_id>/manifest.json`
  with: config, data snapshot hashes, code SHA, seeds, start/end time,
  wall-clock duration, and final metrics.

## 8. Anti-leakage checklist

Before merging any new feature or model, answer in writing:

- [ ] Does this feature at time *t* use only data with timestamp ≤
      *t − publication delay*?
- [ ] If a forecast and an actual both exist, did I use the forecast?
- [ ] Did I fit standardizers / encoders on train only and apply forward?
- [ ] Are my walk-forward windows refitting scalers and models at each step?
- [ ] Is my CV chronological? No shuffle, no K-fold?
- [ ] Did I keep the test set untouched?

If any answer is "not sure," do not run the experiment.
