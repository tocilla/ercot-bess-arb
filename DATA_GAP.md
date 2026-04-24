# Data gap — what we can get, what we can't, what it costs

Living document. The session so far has been limited to `gridstatus.Ercot`
(which wraps older MIS endpoints) and so under-estimated what's available.
This document reconciles the gap and names concrete next sources.

## Status update — 2026-04-24

Infrastructure for every source except commercial ones is now in the
repo. The fetchers are thin, cached, tested:

| Source               | Module                       | Status                              |
|----------------------|------------------------------|-------------------------------------|
| ERCOT RTM SPP        | src/data/ercot.py            | live, 14 years cached               |
| ERCOT historical load| src/data/ercot_load.py       | live, 14 years cached               |
| FRED gas prices      | src/data/fred.py             | live, Henry Hub 1997+ cached        |
| EIA-930              | src/data/eia930.py           | live, 2024 smoke-tested             |
| HRRR weather         | src/data/hrrr.py             | live, single-cycle smoke-tested     |
| ERCOT Public API     | src/data/ercot_api.py        | scaffolded, blocked on u/p          |

Run `python scripts/smoke_data_sources.py` to re-verify them all.

The only remaining credential gap is ERCOT account username+password for
the Public API OAuth flow. Subscription key is already wired in `.env`.
Once those two lines are filled, NP4-732-CD / NP4-737-CD / NP3-560-CD /
NP3-233-CD all become available.

## Status summary

| Source                                     | Status for our project | Cost            | Effort     |
|--------------------------------------------|------------------------|-----------------|------------|
| ERCOT RTM SPP (prices)                     | ✅ have, 2011–2024     | free            | done       |
| ERCOT historical actual load by zone       | ✅ have, 2011–2024     | free            | done       |
| ERCOT load forecast (vintaged STLF)        | ❌ missing → **can get** via ERCOT Public API | free | 1–2 days |
| ERCOT wind actual + forecast (vintaged STWPF) | ❌ missing → **can get** via ERCOT Public API | free | 1–2 days |
| ERCOT solar actual + forecast (vintaged STPPF) | ❌ missing → **can get** via ERCOT Public API | free | 1–2 days |
| ERCOT resource outage capacity             | ❌ missing → **can get** via ERCOT Public API | free | 1 day  |
| EIA-930 hourly demand + forecast + gen-by-fuel | ❌ missing → **can get** via EIA API | free  | 0.5–1 day |
| ERA5 weather reanalysis (historical actuals) | ❌ missing → **can get** via Copernicus CDS (free, key required) | free | 1 day |
| **HRRR historical weather *forecasts* (as-of)** | ❌ missing → **can get** from AWS S3 (anonymous, NO KEY) | free | 1 day |
| NOAA NDFD historical weather *forecasts*   | ✅ on AWS S3 (anonymous, NO KEY) — GRIB2 parse | free | 2 days |
| Natural gas prices (Henry Hub, Waha daily) | ❌ missing → **can get** via FRED CSV (NO KEY) or EIA API | free | 0.5 day |
| GridStatus.io hosted API (backfilled)      | ⚠️ easier drop-in for most of the above       | free tier + paid | 0.5 day |

**The punchline: most of what we thought was missing is actually free with
signup.** The gate was not the data — it was which endpoints gridstatus
wrapped.

---

## 1. What we missed — the ERCOT Public API

Key finding: ERCOT migrated from the old MIS archive (which rolls off in
~8 days) to a **Public API with ≥ 7 years of history**. Requires a free
account at `https://apiexplorer.ercot.com/`. Returns JSON / CSV per
report ID.

Relevant endpoints for our use case:

| Report ID    | Description                                          |
|--------------|------------------------------------------------------|
| NP4-732-CD   | Wind Power Production — hourly actual + STWPF + WGRPP |
| NP4-737-CD   | Solar Power Production — hourly actual + STPPF       |
| NP3-560-CD   | Seven-Day Load Forecast by Model and Weather Zone    |
| NP3-233-CD   | Hourly Resource Outage Capacity                      |
| NP4-742-CD   | Hourly Aggregated Wind Output                        |
| NP1-346-ER   | Unplanned Resource Outages (forced + maintenance)    |

These are **vintaged** — each report has a publish timestamp, so you can
select only records available at time *t*. This is exactly what
METHODOLOGY §3 demands for leakage-free features.

gridstatus has a beta `gridstatus.ercot_api.ercot_api` module that wraps
some of these; it's different from the `gridstatus.Ercot` class we've
been using. Drop-in but needs an API key configured in env.

**Action:** register, add `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`,
`ERCOT_API_SUBSCRIPTION_KEY` to `.env`, extend `src/data/` with an
`ercot_api.py` module for each report.

## 2. EIA-930 — covers every US ISO uniformly

EIA collects Form EIA-930 from every US balancing authority — hourly
demand, hourly demand *forecast*, hourly net generation *by fuel type*,
interchange. For ERCOT, coverage starts **2015-07-01** for original
elements and **2018-07** for fuel-mix breakdowns. Free access via the
EIA Open Data API.

What this gives us that we don't have:
- Day-ahead demand forecast from ERCOT itself (D-1 publish time), as a
  vintaged history — exactly the "what did the operator expect" feature.
- Generation by fuel (wind, solar, gas, coal, nuclear) hourly — a proxy
  for renewable output and capacity margin even without native wind/
  solar actuals.
- Net interchange (not particularly relevant for ERCOT which is almost
  islanded, but useful for CAISO/PJM comparisons later).

**Action:** register for EIA API key, add `src/data/eia930.py` to pull
and cache CSVs. One route for all US ISOs — cheaper than per-ISO work
if we ever port the pipeline.

## 3. ERA5 — free hourly weather reanalysis back to 1940

ERA5 is ECMWF's reanalysis, ~31 km grid, hourly, global. Free through
the Copernicus Climate Data Store after signing up for a personal
access token. Python client `cdsapi`. Variables relevant to us:

- `2m_temperature` — heat-wave proxy, single biggest scarcity driver
- `10m_u_component_of_wind`, `10m_v_component_of_wind` — wind resource
- `100m_u_component_of_wind`, `100m_v_component_of_wind` — closer to
  hub-height, more relevant for wind generation
- `surface_solar_radiation_downwards` — solar resource
- `surface_pressure`, `2m_dewpoint_temperature` — humidity / humidex

Coverage area: a Texas bounding box of ~25–37°N, 93–107°W at 0.25°
resolution is a modest download (tens of MB per year per variable).

**Important caveat — ERA5 is a REANALYSIS, not a forecast.** Values at
time *t* are produced using observations from around *t* (including
some after *t*) — so they are not what a forecaster at time *t* would
have seen. For honest walk-forward, ERA5 must be treated with a lag
(e.g. yesterday's temperature as a proxy for today's weather
conditions), not as a live feature. For a TRUE vintaged forecast
history, use HRRR or NDFD from S3 (below) — free, no API key, and
archived by cycle so you get exactly "what was the 15:00 Houston
temperature forecast as of the 06:00 run".

**Action:** register at Copernicus CDS, add `src/data/era5.py` with a
cdsapi wrapper. Download Texas bbox per year, save as NetCDF/zarr.

## 4. NOAA model archives on S3 — true vintaged forecasts, no API key

**HRRR (High-Resolution Rapid Refresh)** is the right primary source
for ERCOT work. 3 km CONUS grid, hourly cycles, 18-hour forecast
horizon (48h on the 00/06/12/18 cycles), archive since 2014. On AWS
S3 as `noaa-hrrr-bdp-pds` — anonymous access, no API key, no cost.
Key layout is exactly what you want: `<param>/<vertical>/YYYYMMDD/HHZ/FFF.grib2`
so "what did the 06Z run predict for hour 09" is a direct S3 key.

**GFS (Global Forecast System)** — 0.25° global, 4 cycles/day, forecast
out to 384h. Bucket `noaa-gfs-bdp-pds`. Anonymous. Useful as backstop
when HRRR misses a window.

**NDFD (National Digital Forecast Database)** — NWS-published forecast
grids (what gets pushed to local weather offices). 2008+. AWS S3 and
FTP. Anonymous. Useful for longer-horizon features than HRRR.

**Files are GRIB2.** Use `cfgrib` + `xarray`, or `pygrib`, to parse.
For our use case, the per-cycle workflow is:
  1. Pick cycle time (e.g. 06:00 CT — after ERCOT DAM closes).
  2. Extract forecast hours F+3 … F+30 for the Texas bbox.
  3. Aggregate to hourly per weather zone.
  4. Persist to parquet keyed by `(publish_cycle, valid_time)`.

Effort: ~1 day for HRRR alone (S3 + GRIB2), ~2 days with NDFD
alongside. All free.

**Action:** make HRRR the primary weather feature source. ERA5 stays
as an "actuals" calibration against HRRR forecast error.

## 5. Natural gas — cheap signal, easy to add

Daily Henry Hub spot and regional hub differentials (Waha, MidCon, etc.)
are the marginal-fuel economics that drive ERCOT thermal dispatch. Free
from EIA's natural gas data pages and mirrored on FRED. Daily granularity
is fine for a feature that changes slowly.

**Action:** add `src/data/natural_gas.py` that pulls from FRED
(`DHHNGSP` daily series) or EIA `RNGWHHD` series.

## 6. GridStatus.io — paid shortcut

If engineering time is the bottleneck rather than money, GridStatus.io's
hosted API offers all the backfilled ERCOT forecasts (wind, solar, load)
as datasets, pre-parsed, through one consistent Python client
(`gridstatusio`). Free tier: 1M rows/month. Paid tiers for production use.

This would collapse sections 1–2 into an afternoon of integration. Worth
considering for a serious push but not necessary for phase 2.

## 6b. Registration and access summary

**Free registration — email and instant-ish key:**

| Source              | Register at                                                          |
|---------------------|-----------------------------------------------------------------------|
| ERCOT Public API    | https://apiexplorer.ercot.com/                                        |
| EIA Open Data API   | https://www.eia.gov/opendata/register.php                             |
| Copernicus CDS (ERA5) | https://cds.climate.copernicus.eu/how-to-api                        |
| GridStatus.io       | https://www.gridstatus.io/pricing                                     |
| FRED                | https://fred.stlouisfed.org/docs/api/api_key.html                     |

**No registration at all — anonymous HTTPS / S3:**

| Source                        | Access                                                   |
|-------------------------------|----------------------------------------------------------|
| NOAA HRRR forecasts           | `s3://noaa-hrrr-bdp-pds` anonymous read                  |
| NOAA GFS forecasts            | `s3://noaa-gfs-bdp-pds` anonymous read                   |
| NOAA NDFD forecasts           | AWS Registry of Open Data, anonymous                     |
| EIA bulk CSVs / ZIPs          | direct HTTPS, no key needed                              |
| FRED CSV downloads            | `fredgraph.csv?id=<series>` direct HTTPS                 |
| ERCOT MIS historical CSVs     | direct HTTPS (what gridstatus uses for prices + load)    |
| ERCOT historical load ZIPs    | `https://www.ercot.com/gridinfo/load/load_hist`          |

**Notes on the no-key paths:**
- For S3, use `aws s3 cp --no-sign-request` or `s3fs`/`boto3` with
  `config=botocore.UNSIGNED` — no AWS account required.
- FRED's CSV URL pattern (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP`)
  returns plain CSV and is robust enough to script against.
- EIA's bulk paths are documented at `https://www.eia.gov/opendata/bulkfiles.php`.

## 7. What we genuinely can't get (cheaply)

- **Live, sub-hourly operator decisions** — the commitment/dispatch
  signals an ERCOT market participant sees internally. Requires paid
  market-participant credentials. Not happening.
- **Individual generator bids** — offers and commitment prices by unit.
  Some are publicly released with 60-day lag (NP3-970-EX etc.) but
  attribution to specific units is obfuscated.
- **Commercial fuel forecasts** — NOAA NWP isn't the same as a private
  trader's blended weather ensemble. Vendors like DTN, Atmospheric G2,
  Radiant Solutions sell these.
- **Intraday ERCOT-internal forecast revisions** — the SCED/RAS
  intermediate forecasts. Partially in NP4-442-CD but not all models.

## Revised ranking for phase 2

Ordered by expected revenue-lift-per-engineer-day:

1. **ERCOT Public API — wind + solar forecasts (NP4-732-CD, NP4-737-CD).**
   Direct signal on the scarcity-driving capacity margin. ~1–2 days.
2. **ERCOT Public API — 7-day load forecast by zone (NP3-560-CD).**
   Vintaged D-1 load forecast, strictly better than our lagged actuals.
   ~0.5 day.
3. **EIA-930 gen-by-fuel.** Adds a simple renewable-penetration feature
   that generalizes to other ISOs. ~0.5 day.
4. **Natural gas prices.** Cheap to add, real information content for
   thermal-dispatch marginal cost. ~0.5 day.
5. **HRRR weather forecasts from S3 (NO KEY).** Vintaged temperature,
   wind at 80m, solar — the honest weather-forecast signal a live
   operator would have had. ~1 day for GRIB2 setup + Texas bbox.
6. **ERCOT resource outage capacity (NP3-233-CD).** Complement to the
   capacity margin story. ~0.5 day.
7. **ERA5 actuals as calibration.** Only after HRRR is in — useful
   for measuring HRRR's forecast error as a feature, not for
   replacing HRRR. ~1 day.

## Expected ceiling if we executed all of #1–#6

We don't actually know. The session's working estimate, based on the
literature: with a vintaged day-ahead forecast of load + wind + solar +
temperature, well-tuned ML models report 65–80% of perfect-foresight
revenue on ERCOT BESS arbitrage with 1 cycle/day. That would close most
of the gap between our current 53% and the 87% floor — possibly even
beat the floor by explicitly avoiding cycles on unprofitable days a
perfect-timing oracle would also avoid.

This remains an open empirical question and is the natural scope of the
next project phase.

## Open items for this document

- [ ] Confirm ERCOT API rate limits + historical cap.
- [ ] Confirm EIA-930 fuel-mix start date for ERCOT specifically.
- [ ] Confirm NDFD availability for our validation window 2020-11+.
- [ ] Scope a concrete DATA_BACKFILL milestone in [PLAN.md](PLAN.md).
