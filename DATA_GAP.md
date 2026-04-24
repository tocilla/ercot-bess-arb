# Data gap — what we can get, what we can't, what it costs

Living document. The session so far has been limited to `gridstatus.Ercot`
(which wraps older MIS endpoints) and so under-estimated what's available.
This document reconciles the gap and names concrete next sources.

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
| ERA5 weather reanalysis (historical actuals) | ❌ missing → **can get** via Copernicus CDS | free | 1 day |
| NOAA NDFD historical weather *forecasts*   | ⚠️ available via NCEI AIRS but heavy          | free | 2–3 days |
| Natural gas prices (Henry Hub, Waha daily) | ❌ missing → **can get** via EIA / FRED API | free | 0.5 day |
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
conditions), not as a live feature. For a TRUE forecast history, use
NDFD (below).

**Action:** register at Copernicus CDS, add `src/data/era5.py` with a
cdsapi wrapper. Download Texas bbox per year, save as NetCDF/zarr.

## 4. NDFD — historical NWS forecast archive

NOAA's National Digital Forecast Database archives the actual forecast
grids the NWS published. Pre-2008 data requires the AIRS archive; post-
2008 via S3 (`registry.opendata.aws/noaa-ndfd/`) and FTP. Format: GRIB2.
Variables: temperature, wind, sky cover, precipitation probability.

Unlike ERA5, NDFD contains true vintaged forecasts — "what did the
weather service think about tomorrow, when yesterday at 6am?". This is
the closest free analogue to the operator-side weather information.

Effort: higher than ERA5. GRIB2 parsing, S3 key schema, temporal
indexing. Plan on 2–3 days to integrate cleanly.

**Action:** park for phase 3. Useful once we've validated that ERA5
actuals add value.

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
5. **ERA5 temperature + wind.** Heat-wave and wind-ramp features. ~1 day.
6. **ERCOT resource outage capacity (NP3-233-CD).** Complement to the
   capacity margin story. ~0.5 day.
7. **NDFD historical forecasts.** Only if we first prove ERA5 actuals
   add value. ~2–3 days.

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
