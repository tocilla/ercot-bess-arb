# ERCOT BESS arbitrage — forecast-driven dispatch research

A research investigation into how much of the theoretical arbitrage revenue
a 100 MW / 200 MWh battery in ERCOT can realistically capture using
forecast-driven dispatch, evaluated under proper walk-forward discipline
on a single-shot held-out test set.

## Headline result (test window, 2023-01-01 → 2024-12-31)

**The deployable claim:** the model adds **+$2.6M/year** of arbitrage revenue
on a 100 MW battery over a persistence baseline — **+26.3 pp** vs. the most
relevant deployable comparison.

**The academic-comparable claim:** the model captures **77.13%** of the
perfect-foresight revenue ceiling — within the literature's typical range
of 50–70% for 1-cycle/day arb on daily-vintage features, below commercial
production claims (Ascend Analytics: 90–95% with full-stack DAM+RTM+ancillary
joint optimization — a wider problem than this).

| Strategy | Revenue | $/kW-yr | Δ vs persistence | % of ceiling |
|---|---:|---:|---:|---:|
| Do nothing | $0 | 0.0 | −$10.24M | 0.00% |
| Fixed time of day (3am→5pm) | $3.10M | 15.5 | −$7.14M | 15.42% |
| Persistence forecast | $10.24M | 51.2 | (baseline) | 50.88% |
| Seasonal-naive forecast | $10.60M | 52.9 | +$0.35M | 52.63% |
| **This project's ML ensemble** | **$15.53M** | **77.6** | **+$5.29M** | **77.13%** |
| Natural-spread floor (oracle, *not deployable*) | $19.12M | 95.5 | +$8.87M | 94.95% |
| LP ceiling (perfect foresight, *theoretical bound*) | $20.13M | 100.6 | +$9.89M | 100.00% |

> **Note on floor and ceiling.** The bottom two rows are **not strategies you
> can run** — they cheat by using realized prices that wouldn't have been
> knowable at decision time. The natural-spread floor sorts each day's
> *realized* prices and picks the cheapest/most-expensive intervals; the LP
> ceiling solves a perfect-foresight optimization given every future price.
> Both are *benchmarks* that bound how big the arbitrage pie is and how
> much of it is theoretically reachable. The deployable comparison is
> persistence — that's why the headline `Δ vs persistence` claim is
> measured against it, not against the floor or the ceiling.

Full results, regime breakdown, and industry context: [RESULTS.md](RESULTS.md).

## Project structure

```
.
├── PLAN.md          # project charter (what / why / how)
├── METHODOLOGY.md   # evaluation protocol — splits, walk-forward, metrics
├── DATA.md          # data sources, as-of semantics, schema, references
├── FINDINGS.md      # running log of every experiment with TL;DR at top
├── DECISIONS.md     # architecture / scope decisions with rationale
├── RESULTS.md       # final test-set numbers
├── src/             # battery sim, dispatch, baselines, features, models
├── scripts/         # entry points (data fetch, experiments, reveal)
├── tests/           # 31 unit tests incl. anti-leakage tests
├── configs/         # YAML configs (splits, battery spec)
└── data/, results/  # gitignored: raw data, model artifacts
```

## What's in scope

- ERCOT RTM Settlement Point Prices at HB_NORTH, 2011–2024 (15-min)
- 1 cycle/day cap, 100 MW / 200 MWh / 85% round-trip battery
- Daily-vintage exogenous features (load, EIA-930, weather, ERCOT forecasts)
- Walk-forward evaluation with monthly retraining
- Threshold-rule + forecast-gate dispatch (no MILP, no DAM/RTM split)

## What's not in scope

- Ancillary services (RegUp/RegDown, ECRS) — typically 50–70% of real-world
  BESS revenue, but a different problem
- Sub-15-min dispatch
- Co-located solar / hybrid configurations
- Real-time generator outage feeds (likely the biggest remaining alpha source)
- Live deployment — this is research, not a product

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fill in API keys you have. ERCOT and EIA are free-with-signup; others have
# anonymous or paid alternatives.
cp .env.example .env
# Edit .env with your keys; never commit it.
```

## Reproducing the headline result

```bash
# 1. Cache RTM prices (~5 min, anonymous)
python scripts/fetch_ercot_rtm.py --start 2011 --end 2024

# 2. Cache historical load (~3 min, anonymous)
python scripts/fetch_ercot_load.py --start 2011 --end 2024

# 3. Cache EIA-930 demand+gen-mix (~5 min, free EIA key)
python scripts/fetch_eia930_history.py --start 2019 --end 2024

# 4. Run the test-set reveal (~12 min on a laptop)
python scripts/test_reveal.py
```

The walk-forward harness re-fits the model at every 30-day boundary inside
the test window, so the run is multi-minute regardless of whether anything
is cached.

## Environment variables

See [.env.example](.env.example) for the full template.

| Module | Required env vars |
|---|---|
| ERCOT RTM SPPs (prices) | none |
| ERCOT historical load | none |
| EIA-930 (demand + gen-mix) | `EIA_API_KEY` |
| FRED (gas prices) | none |
| NOAA HRRR (weather forecast) | none (anonymous S3) |
| ERCOT Public API (vintaged forecasts) | `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`, `ERCOT_API_SUBSCRIPTION_KEY` |

`.env` is gitignored. Never paste keys in issues, PRs, or chat. Regenerate if you do.

## Methodology highlights

- **Chronological 70/15/15 split.** Train 2011-2020-10, val 2020-11-2022-12,
  test 2023-01-2024-12. No random sampling. See [configs/splits.yaml](configs/splits.yaml).
- **Test set touched once.** Every model selection, hyperparameter, and feature
  decision was made on val. Test was revealed after the spec was frozen.
- **Walk-forward retraining** at 30-day boundaries. Each retrain uses only
  data strictly before the boundary.
- **Anti-leakage tests** verify no future data influences current-time features
  (see [tests/test_features_and_walk_forward.py](tests/test_features_and_walk_forward.py)).
- **Multi-seed evaluation.** ±2.84 pp seed std on val measured before any
  comparative claims, so deltas are validated against noise.

## Notable findings on the way to the headline

Documented in [FINDINGS.md](FINDINGS.md):

- **Truncated training (2019+) beat full-history (2011+) with NaN exogenous
  features.** Empirically, by 5.8 pp on val. ERCOT's pre-2019 market
  (pre-winterization, less wind/solar) is a different distribution.
- **Daily-vintage HRRR temperature forecasts hurt dispatch revenue** while
  marginally improving MAE — first of several "MAE down, revenue down"
  observations in the session.
- **ERCOT STWPF/STPPF (operator-grade wind/solar forecasts) tightened seed
  variance 3× but did not move mean revenue.**
- **Decision-aware loss weighting hurt dispatch on every variant tried.**
  Mechanistic reason: the threshold dispatch ranks intervals within each
  day, so any loss-shape that distorts central-distribution calibration
  corrupts the rank ordering.
- **Seed ensemble adds +5 pp on val (+1.14 pp on test) for free** by
  averaging predictions from 5 seeded fits.

## License

[MIT](LICENSE).

## Things you should NOT trust this project for

- A deployable, real-time BESS dispatch system. This isn't engineered for
  production.
- Tomorrow's price prediction. The forecaster is good enough to inform
  daily charge/discharge selection but not for prop trading or hedging.
- Generalization to other ISOs. CAISO, PJM, etc. would need different feature
  sets and likely different splits.
- Short-duration batteries. The setup is tuned for a 2-hour battery.
