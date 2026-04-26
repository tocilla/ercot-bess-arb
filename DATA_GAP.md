# Data landscape — what we tried, decided against, missed

> **About this document.** Originally a planning doc written mid-project
> to inventory data sources. Now a post-project reference — what was
> tried, what was deliberately skipped, and what we genuinely missed.
> Useful for anyone doing similar BESS-arb research who wants to know
> what's free, what needs an API key, and what's genuinely paywalled.
> For the project's results see [RESULTS.md](RESULTS.md); for the
> experiment log see [FINDINGS.md](FINDINGS.md).

## What we tried (tested in experiments)

| Source | In final spec? | What we observed |
|---|---|---|
| ERCOT RTM SPP (prices) | **yes** — primary target | 2011–2024 cached; baseline of the whole problem |
| ERCOT historical load by zone | **yes** — feature | 2011–2024 cached |
| EIA-930 demand actual + DA forecast + gen-by-fuel | **yes** — feature | 2019–2024 cached; included in final 27-feature spec |
| ERCOT Public API wind STWPF (NP4-732-CD) | **no** — tested, neutral | Δ −1.79 pp on val, within noise (−0.94σ). Variance shrunk 3× but mean unchanged. |
| ERCOT Public API solar STPPF (NP4-737-CD) | **no** — tested, neutral | Same as wind |
| HRRR weather forecasts (1 cycle × 1 fxx/day) | **no** — tested, neutral | First-look −8.5 pp single-seed, later shown to be within seed noise (±2.84 pp on val) |
| Decision-aware loss weighting (3 schemes) | **no** — tested, hurt | Every variant lost decisively to uniform weights. Mechanistic reason: dispatch ranks intervals, weighted loss distorts rank. |

Per the documented decision rule, exits from "tested-and-rejected" entries
were taken when the lift was below 2σ of seed noise.

## What we deliberately skipped (and why)

| Source | Why skipped |
|---|---|
| ERCOT 7-day load forecast (NP3-560-CD) | Pattern-similar to wind/solar STWPF/STPPF — daily-vintage, smooth quantity, broadcast across 96 intraday intervals. Wind/solar's negative result strongly suggested this would behave the same. |
| ERA5 weather reanalysis | Reanalysis (not vintaged) → has to be used as a lagged proxy, similar information to what HRRR already gave us. |
| NOAA NDFD historical forecasts | Heavier integration (GRIB2 + AWS). Pattern-similar to HRRR which we already tested. |
| Hourly-vintage ERCOT forecasts (24× more docs) | Cost is high (~hours of API time), and the daily version showed nothing — prior was that hourly wouldn't unlock a different result. |
| FRED Henry Hub natural gas as a feature | Slow-moving signal, unlikely to drive intraday rank. Fetched but never integrated. |

These were all skipped because the documented decision rule said
"if the first daily-vintage forecast feature doesn't help → exit the
data-gathering branch and write up." The rule treated all remaining
items as one bucket, which was a coarse simplification.

## What we genuinely missed (and have since tested post-hoc)

**ERCOT outage capacity (NP3-233-CD).** Originally flagged as the one
untested signal that might behave differently because outages are
*discrete events* rather than smooth daily forecasts. Tested
post-hoc on 2026-04-26 (val window only — see FINDINGS):

  baseline (EIA only):  57.67 ± 2.84 pp of ceiling
  + outage capacity:    55.34 ± 3.09 pp of ceiling
  Δ:                    −2.33 pp at −0.78σ

**Within seed noise but trending negative — same pattern as wind/solar/HRRR.**
The "different signal type" hypothesis didn't pan out. Mechanism: the
forecast-gate dispatch ranks intervals within each day; broadcasting a
daily aggregate outage value across 96 intraday intervals shifts the
predicted level uniformly without changing rank order. Same failure
mode as smooth forecasts.

This **strengthens** the central diagnosis: daily-vintage exogenous
data fundamentally doesn't help rank-based dispatch in this framing,
regardless of signal type. Closing the residual gap to floor would
require **hourly-vintage** feeds and/or a dispatch formulation that
uses level signals (e.g. continuous bid curves into DA/RT markets) —
both scope-shifts beyond this project.

**For a future extender**, the still-untested but-pattern-similar
items (load forecast NP3-560-CD, ERA5, NDFD) are unlikely to change
the picture. The actionable path forward is fetching hourly-vintage
forecasts (24× more docs each) or moving to a multi-market dispatch
framing — not more daily-aggregate features.

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

## What we tested and learned (after publishing)

The phase-2 ranking above was a forward-looking plan. After running
the experiments documented in [FINDINGS.md](FINDINGS.md):

- **EIA-930 demand + gen-mix:** added as features. Lift on val, kept in
  the final spec. Final test: 77.13% of perfect-foresight ceiling.
- **ERCOT STWPF + STPPF (wind+solar forecasts):** measured neutral on
  multi-seed val (Δ −1.79 pp, within −0.94σ). Variance shrunk 3× but
  mean unchanged. Did not include in final spec.
- **HRRR weather (1 cycle/day, F+6):** measured −8.5 pp single-seed,
  later shown to be within seed noise (±2.84 pp). Not in final spec.
- **Decision-aware loss weighting:** every variant lost decisively to
  uniform weights (mechanism in FINDINGS — dispatch ranks intervals,
  weighted loss distorts rank).

So the headline answer to "would more daily-vintage exogenous data
close the gap to floor (95% on test)?" is **probably not** — daily
aggregate forecasts don't help a rank-based forecast-gate dispatch.
The remaining ~18 pp of headroom likely requires **hourly-vintage**
forecasts, generator-level outage data, or a fundamentally different
dispatch formulation (decision-focused learning would also need to
solve the rank-distortion problem the weighted-loss experiment
exposed).

## Open items if extending the project

- [ ] Hourly-vintage ERCOT forecasts (24× more docs per endpoint)
- [ ] NP3-560-CD load forecast — last untested vintaged ERCOT signal
- [ ] NP3-233-CD outage capacity — only data source we tested that
      could plausibly encode tail-risk in a way wind/solar don't
- [ ] HRRR with multiple fxx values per day — addresses the
      "daily aggregate is too coarse" concern
- [ ] ERA5 actuals as calibration / auxiliary feature
