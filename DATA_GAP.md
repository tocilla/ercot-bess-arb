# Data gap — reference doc on the ERCOT BESS data landscape

> **About this document.** This was originally a planning doc written
> mid-project to inventory the data sources for ERCOT BESS arbitrage
> research and grade their accessibility. After publishing the project,
> I'm keeping it as a *reference* — useful for anyone doing similar work
> who wants to know what's free, what needs an API key, and what's
> genuinely paywalled. The current-status table below reflects what
> was actually fetched and used. Forward-looking suggestions are
> labeled. For the project's actual results see [RESULTS.md](RESULTS.md);
> for the experiment log see [FINDINGS.md](FINDINGS.md).

## Final status — what was used in the published model

| Source                                     | In the model? | Cached range | Notes |
|--------------------------------------------|---------------|--------------|-------|
| ERCOT RTM SPP (prices)                     | **yes** — primary signal | 2011–2024 | anonymous |
| ERCOT historical actual load (by zone)     | **yes** — feature | 2011–2024 | anonymous |
| EIA-930 demand + DA forecast + gen-by-fuel | **yes** — feature | 2019–2024 (ERCO availability) | needs free `EIA_API_KEY` |
| ERCOT Public API wind STWPF (NP4-732-CD)   | tested, did not lift | 2019–2022, 1 doc/day | needs free ERCOT account + sub key |
| ERCOT Public API solar STPPF (NP4-737-CD)  | tested, did not lift | 2019–2022, 1 doc/day | needs free ERCOT account + sub key |
| HRRR weather forecasts (NOAA, AWS)         | tested, did not lift | 2019–2022, 1 cycle × 1 fxx/day | anonymous S3 |
| FRED Henry Hub gas spot                    | fetched but not added as feature | 1997–present | anonymous |
| ERCOT 7-day load forecast (NP3-560-CD)     | not fetched | — | needs free ERCOT credentials |
| ERCOT outage capacity (NP3-233-CD)         | not fetched | — | needs free ERCOT credentials |
| ERA5 weather reanalysis (Copernicus)       | not fetched | — | free with CDS key |
| NOAA NDFD historical NWS forecasts         | not fetched | — | anonymous S3, GRIB2 |
| GridStatus.io hosted API                   | not used | — | paid drop-in shortcut |

`scripts/smoke_data_sources.py` re-verifies each fetcher end-to-end.

**The take-away:** for ERCOT BESS-arb research, you can build the entire
data side of a project like this with free credentials only — ERCOT
Public API + EIA + Copernicus + NOAA S3 cover almost everything an
industry-grade research model needs except generator-level outage data
and proprietary weather ensembles.

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
