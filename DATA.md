# Data

Data sources, as-of semantics, publication delays, revisions, and schema.
This is the most accident-prone part of the project — read carefully before
adding any feature.

## 1. Target ISO (to be decided)

See [DECISIONS.md](DECISIONS.md). Leaning ERCOT; CAISO is the alternative.

## 2. Sources

### 2.1 ERCOT
- **RTM SPPs (Real-Time Market Settlement Point Prices)** — **15-minute**
  granularity (one price per 15-min settlement interval, not 5-min). This is
  the price a battery operator actually gets paid; 5-min SCED LMPs are
  dispatch signals, not settlement prices. Source: ERCOT MIS report
  NP6-785-ER. Historical archive begins 2011-01-01. Covers 7 hubs + 8 load
  zones. Accessed via `gridstatus.Ercot.get_rtm_spp(year=YYYY)` and cached
  to `data/raw/ercot/rtm_spp_<year>.parquet`.
- **DAM LMPs (Day-Ahead Market)** — hourly. Cleared ~10:30 CPT day before.
- **Load (system-wide and by zone)** — 5-min actual, published with lag.
  Also day-ahead load *forecast* — use this as a training feature, not actual.
- **Wind / solar generation, actual** — post-hoc, with revisions.
- **Wind / solar generation, forecast** — STWPF / STPPF, published pre-operation.
  **This is the feature to use, not the actual.**
- **Outage / capacity reports** — daily publication.

### 2.2 CAISO (alternative)
- OASIS API. LMPs (RTM 5-min, DAM hourly), load, renewable generation.
- Similar forecast-vs-actual distinction applies.

### 2.3 Weather
- **ERA5 reanalysis** (ECMWF / Copernicus) — historical, post-hoc. Useful as
  ground-truth *only*; never use as a feature at inference time because it
  would not have been available.
- **NWS / NOAA forecasts** — use the forecast available at decision time, not
  the forecast later revised with hindsight.
- Variables: temperature, wind speed at 10m/100m, solar irradiance, cloud cover.

## 3. Publication delays (the leakage landmine)

For every feature used at time *t*, only values with effective timestamp
≤ *t − d* are admissible, where *d* is the publication delay. Record *d*
explicitly per feature in `configs/features.yaml`.

Starter table (to be verified against current ISO documentation before use):

| Feature                          | Granularity | Publication delay (approx) |
|----------------------------------|-------------|----------------------------|
| ERCOT RTM SPP (settlement price) | 15 min      | ~15–30 min after interval  |
| ERCOT SCED LMP (dispatch signal) | 5 min       | ~5–10 min after interval   |
| ERCOT DAM LMP                    | hourly      | available ~10:30 CPT D−1   |
| ERCOT system load actual         | 5 min       | ~5–15 min                  |
| ERCOT system load DA forecast    | hourly      | available ~D−1 morning     |
| ERCOT wind STWPF forecast        | hourly      | refreshed multiple times/day |
| ERCOT solar STPPF forecast       | hourly      | refreshed multiple times/day |
| ERCOT wind/solar actual          | 5 min       | ~5–15 min, revised later   |
| Weather NWS forecast             | hourly/6hr  | issued at fixed cycles     |
| ERA5 reanalysis                  | hourly      | several days to months     |

**Do not trust this table blindly.** Verify current values against source
documentation and record citations in `configs/features.yaml`.

## 4. Revisions

ISO data is revised post-hoc. Prices can be restated during settlement;
renewable "actuals" often differ between first publication and the final
archived value.

- Store each feed with an **as-of** timestamp: when was this value first
  published? Pull from the archival feed if available; otherwise snapshot on
  ingest and never overwrite.
- Training features at time *t* must use the version that was public at
  *t − d*, not the final revised value in today's archive.
- If as-of history is not available for a source, document the limitation in
  [FINDINGS.md](FINDINGS.md) and treat any result that depends on that feature
  as an upper bound (optimistic).

## 5. Storage layout

- `data/raw/` — exactly as downloaded. Never edited. Filename includes source
  and download date: `ercot_rtm_lmp_HB_HOUSTON_2015-2024_asof_20260424.csv.gz`.
- `data/interim/` — parsed, schema-normalized, but not feature-engineered.
  Parquet format.
- `data/processed/` — feature sets ready for modeling, one parquet per
  experiment/config.

A `data_manifest.json` in each directory records SHA256, row count, and
source for every file. The manifest is checked into git; the files are not.

## 6. Time zone discipline

- All data stored in UTC internally. Convert to local tz (America/Chicago for
  ERCOT, America/Los_Angeles for CAISO) only for presentation.
- DST transitions are a real bug source in ISO data (missing and duplicate
  hours). Validate every new feed against expected row count per day.

## 7. Schema conventions

Processed feature tables use:

- `timestamp_utc`: tz-aware UTC timestamp, interval *start*.
- `node`: settlement point / node identifier.
- Feature columns: snake_case, suffixed with horizon when applicable
  (e.g. `load_forecast_h4` = load forecast for 4 hours ahead, as known at
  `timestamp_utc`).
- Target column: `lmp_rtm` (5-min) or `lmp_dam` (hourly).

## 8. Open data questions

- Does ERCOT publish a historical "as-of" archive for STWPF / STPPF forecasts,
  or do we have to start our own snapshotting?
- What's the earliest usable date? ERCOT nodal market started 2010-12-01.
- For CAISO, same question: are historical *forecasts* retrievable or only *actuals*?
