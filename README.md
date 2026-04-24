# Battery Arbitrage on ERCOT / CAISO

ML system that forecasts short-horizon electricity prices and dispatches a
simulated grid-scale battery to maximize arbitrage revenue.

See [PLAN.md](PLAN.md) for the project charter — goal, scope, baselines,
evaluation protocol, and ML pitfalls to avoid.

## Documents

- [PLAN.md](PLAN.md) — project charter (what / why / how)
- [METHODOLOGY.md](METHODOLOGY.md) — evaluation protocol (splits, walk-forward,
  metrics, revenue attribution). Single source of truth.
- [DATA.md](DATA.md) — data sources, as-of semantics, publication delays, schema.
- [RESEARCH.md](RESEARCH.md) — literature review, benchmarks, open problems.
- [FINDINGS.md](FINDINGS.md) — running log of experiments and results.
- [DECISIONS.md](DECISIONS.md) — log of scope / design decisions with rationale.
- [DATA_GAP.md](DATA_GAP.md) — what data we have, what we can still get, what's hard.

## Repo layout

```
.
├── src/         # library code (features, models, dispatch, evaluation)
├── scripts/     # entry points (fetch data, run experiment, produce report)
├── tests/       # unit and integration tests
├── notebooks/   # exploratory, not part of the pipeline
├── configs/     # experiment configuration files
├── data/        # gitignored: raw / interim / processed
└── results/     # gitignored: model artifacts, metrics, figures
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy the env template and fill in keys where you have them.
cp .env.example .env
# Edit .env in your editor.
```

### Environment variables

See [.env.example](.env.example) for the full template. Summary of what's
required for each data module:

| Module                       | Env vars needed                                                    |
|------------------------------|--------------------------------------------------------------------|
| ERCOT RTM SPPs (prices)      | none (anonymous MIS archive)                                       |
| ERCOT historical load        | none (anonymous)                                                   |
| EIA-930 (demand + gen-fuel)  | `EIA_API_KEY`                                                      |
| FRED (gas prices)            | none (direct CSV endpoint)                                         |
| NOAA HRRR (weather forecast) | none (anonymous S3 — `boto3` with UNSIGNED config)                 |
| ERCOT Public API (forecasts) | `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`, `ERCOT_API_SUBSCRIPTION_KEY` |
| ERA5 (weather actuals)       | `CDSAPI_URL`, `CDSAPI_KEY` — optional, phase 3                     |

`.env` is gitignored. Never paste keys into issues, PRs, or chat
transcripts; regenerate if you do.

## Running

Entry points live in `scripts/`. Example (TBD once implemented):

```bash
# Fetch raw LMPs for a node
python scripts/fetch_lmp.py --iso ercot --node HB_HOUSTON --start 2015-01-01

# Run a baseline and log to FINDINGS.md
python scripts/run_baseline.py --config configs/baseline_natural_spread.yaml
```

## Discipline

Before any experiment, re-read:
- §6 Baselines in [PLAN.md](PLAN.md) — what to beat
- §8 Evaluation protocol in [METHODOLOGY.md](METHODOLOGY.md) — how to measure
- §9 ML pitfalls in [PLAN.md](PLAN.md) — what silently breaks results
