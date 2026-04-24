# Findings

Running log of experiment results, most recent first. Populated as experiments
complete. Every entry is self-contained: someone reading only that entry
should understand what was tried, what was measured, and what it means.

## Entry template

```
## YYYY-MM-DD — <short title>

**Config:** configs/<file>.yaml, commit <short SHA>
**Data:** <node(s)>, <date range>, split <train / val / test cutoffs>

**What:** one paragraph on what was run.

**How:** model class, features, walk-forward cadence, seeds.

**Results:**
| Strategy          | $/MWh/day | % of ceiling | MAE ($/MWh) | Sharpe | Notes |
|-------------------|-----------|--------------|-------------|--------|-------|
| Natural spread    |           |              | —           |        |       |
| Persistence       |           |              |             |        |       |
| Seasonal naive    |           |              |             |        |       |
| Classical + rule  |           |              |             |        |       |
| This run          |           |              |             |        |       |
| Perfect foresight |           | 100%         | 0           |        |       |

**Revenue attribution:**
- Spread: $__ / day
- Forecast skill: $__ / day
- Optimization: $__ / day

**What broke / what surprised me:** honest notes. Features that leaked, bugs
found, regimes where the model misbehaved, seeds that disagreed.

**Next:** concrete next action, referencing a milestone in [PLAN.md](PLAN.md).
```

---

## Log

<!-- Newest entry at the top. -->

### 2026-04-24 — project initialized

Repo scaffolded. No experiments run yet. First milestone is data pipeline
for one ISO, one node, RTM LMPs + load + renewables.
