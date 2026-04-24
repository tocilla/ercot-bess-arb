# In-progress work (to resume)

Last updated: 2026-04-25.

## ERCOT Public API wind + solar forecast backfill (paused)

**State:** 226/1461 wind docs cached, 0/1461 solar. Paused due to aggressive
rate limiting at 0.3s inter-request pacing.

**Cache location:**
- `data/raw/ercot_forecasts/wind/YYYYMMDD.parquet`
- `data/raw/ercot_forecasts/solar/YYYYMMDD.parquet`

Already-cached files are safe — the fetcher skips anything with a parquet
on disk (see `src/data/ercot_forecasts.py::fetch_daily_forecast`), so
restart is resume-friendly and does **not** re-download.

### How to resume

Run from the repo root, in the venv:

```bash
source .venv/bin/activate
python scripts/backfill_ercot_forecasts.py \
    --start 2019-01-01 --end 2022-12-31 \
    --endpoints wind,solar \
    --pause 3.0
```

The `--pause 3.0` flag is a deliberate slowdown; the ERCOT Public API
sends 429s when we pace much faster than ~1 request per 3 seconds.
Expected wall-clock with this pacing: **~2 hours for both endpoints
combined**, compared to effectively never under the old 0.3s pacing.

If you're running it in a different terminal while another Claude
session is live, either terminal is fine — the cache is the source of
truth.

### Why we're doing this

Next ML experiment compares the current session-best model (q50 LGBM +
forecast-gate + EIA-930 features + 2019+ truncated training, 56.4% of
ceiling on val) against the same setup plus `ercot_stwpf_system_wide`
and `ercot_stppf_system_wide` features. Feature extraction is already
wired into `src.features.build_features(..., ercot_wind_forecasts=,
ercot_solar_forecasts=)`.

The rationale: ERCOT's own operational STWPF / STPPF forecasts should
be more directly tied to capacity margin (and therefore scarcity risk)
than the ambient-temperature signal that HRRR's 12Z F06 Texas summary
provided (which did not help — see FINDINGS 2026-04-25).

### When backfill is done

1. Write `scripts/run_ercot_forecasts_experiment.py` — mirror of
   `scripts/run_hrrr_experiment.py`. Should load wind+solar via
   `src.data.ercot_forecasts.load_forecasts()`, build features with
   and without the new columns, compare revenue on val.
2. Run it, log findings, commit.

## Session-best so far

- Floor (14-year history, oracle timing): 87.0% of ceiling
- Best ML (val): **56.4% of ceiling** with EIA-930 features, 2019+
  truncated training. See FINDINGS.md entry 2026-04-24 for details.
- Test set: untouched. Per DECISIONS.md discipline.
