# Research

Literature review and benchmarks. Populated as we read. Every entry cites a
source and states what we're taking from it.

## 1. Electricity price forecasting — classical

*To fill in with:*
- Weron, R. (2014). "Electricity price forecasting: A review of the
  state-of-the-art with a look into the future." Canonical review. Establishes
  benchmark taxonomy (statistical vs. computational intelligence vs. hybrid).
- Typical reported MAE / RMSE ranges for DA forecasting on EU markets —
  useful sanity checks for our numbers.

## 2. Electricity price forecasting — ML / DL

*To fill in with:*
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). "Forecasting
  day-ahead electricity prices: A review of state-of-the-art algorithms, best
  practices and an open-access benchmark." Provides an open benchmark and a
  strong DL baseline (DNN) that many papers compare against.
- Recent transformer / TFT / N-BEATS results on energy markets.

## 3. Battery dispatch / BESS arbitrage

*To fill in with:*
- Sioshansi, R. et al. — PJM arbitrage studies. Order-of-magnitude revenue
  figures for BESS arbitrage in different ISOs.
- Perfect-foresight vs. forecast-driven revenue gap literature.
- Degradation-aware dispatch models (rainflow cycle counting, throughput
  penalties).

## 4. ERCOT / CAISO market mechanics

*To fill in with:*
- ERCOT protocols (nodal market design, SCED, settlement point types).
- CAISO market operations manual.
- Ancillary services structure in each ISO (later-phase reference).

## 5. Decision-focused learning

*To fill in with:*
- Elmachtoub & Grigas — "Smart Predict, then Optimize." Foundational SPO loss.
- Donti, Amos, Kolter — "Task-based End-to-End Model Learning." Differentiable
  optimization layers.
- Relevance for us: phase 5 stretch goal — train a forecaster to minimize
  dispatch regret directly rather than forecast error.

## 6. Benchmarks we want to beat

Target numbers from published work, to be filled in as we collect them. Each
entry should record: (paper, ISO, node/zone, horizon, data years, MAE, RMSE,
revenue if reported, notes on whether it's apples-to-apples with us).

| Paper | ISO | Horizon | Data years | Metric | Value | Our comparable |
|-------|-----|---------|------------|--------|-------|----------------|
| —     | —   | —       | —          | —      | —     | —              |

## 7. Open questions from the literature

- How much of published BESS arbitrage revenue comes from the natural spread
  vs. forecast skill? We should be able to answer this ourselves with our
  attribution (METHODOLOGY §5.3) — most papers don't decompose it.
- Which papers use walk-forward on a decade-plus of data vs. a single test
  year? Filter claims accordingly.
- For papers reporting "% of perfect foresight," what battery spec and cycle
  cost did they assume? These drive the number as much as the forecast.

## 8. Reading queue

*Add papers here before reading, strike through after.*

- [ ] Lago et al. 2021 — open benchmark paper
- [ ] Weron 2014 — review
- [ ] (more to add)
