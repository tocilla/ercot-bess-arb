# Decisions

ADR-lite log of scope, architecture, and methodology decisions. Newest first.
Every entry records: what was decided, alternatives considered, and why.
If a decision is reversed later, add a new entry — do not edit old ones.

## Entry template

```
## YYYY-MM-DD — <short title>

**Decision:** one sentence.

**Alternatives considered:**
- A — pros / cons
- B — pros / cons

**Rationale:** why we picked this, in 2–4 sentences.

**Revisit when:** the trigger that would make us reopen this.
```

---

## Log

### 2026-04-24 — Project scope and initial phase

**Decision:** Phase 1 scope is one ISO, one node, RTM-only, point forecast,
threshold-rule dispatch. Full evaluation protocol (METHODOLOGY.md) applies
from day one.

**Alternatives considered:**
- Start with deep models and distributional forecasts — rejected: complexity
  hides leakage and debugging is harder when we haven't verified the data.
- Skip evaluation protocol to move faster — rejected: the discipline is the
  whole point; sloppy early numbers poison the project.

**Rationale:** Code generation is fast, but understanding data and avoiding
leakage is not. Keep the code surface small until the data is trustworthy and
the natural-spread baseline is measured.

**Revisit when:** natural-spread baseline and at least one classical forecast
+ threshold rule have been honestly evaluated on the validation set.

### 2026-04-24 — ISO choice: to be decided

**Decision:** pending. Leaning ERCOT.

**Alternatives considered:**
- ERCOT: more price volatility → more arbitrage revenue per MW installed; one
  market (energy-only); simpler to model. Cons: fewer publicly archived
  forecast vintages?
- CAISO: cleaner DAM + RTM structure; extensive documentation; strong
  renewable penetration story. Cons: lower spread → lower headline revenue.

**Rationale:** TBD after inspecting data availability and as-of archives for
forecasts.

**Revisit when:** we know whether historical STWPF / STPPF forecasts are
retrievable from ERCOT (and the CAISO equivalent).
