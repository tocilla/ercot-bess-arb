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

### 2026-04-24 — ISO: ERCOT, node: HB_NORTH

**Decision:** Phase 1 uses ERCOT RTM Settlement Point Prices for the
HB_NORTH hub. Data via `gridstatus.Ercot.get_rtm_spp`, archive begins 2011.

**Alternatives considered:**
- ISO — CAISO: cleaner DAM + RTM structure, more documentation. Rejected for
  phase 1: less volatility → lower headline revenue, fewer scarcity events
  to test tail behavior.
- Node — HB_HOUSTON: also a liquid hub with a strong tourism/load profile.
  Rejected for phase 1: HB_NORTH has more published academic baselines and
  captures more of the renewable-driven spread dynamics.

**Rationale:** ERCOT has the most volatile RTM in the US → biggest absolute
arbitrage spreads. HB_NORTH is the canonical reference hub in ERCOT BESS
research, which makes our results comparable to published work. The
15-minute RTM SPP is the price operators are actually settled at, so it's
the correct revenue target (distinct from 5-min SCED dispatch LMPs).

**Revisit when:** we want to test generalization — run the same pipeline on
HB_HOUSTON and HB_WEST, and/or run on CAISO for a lower-volatility
comparison. Also revisit if forecast-vintage archives become a blocker.

### 2026-04-24 — RTM data is 15-min, not 5-min

**Decision:** All dispatch simulations use 15-minute intervals.

**Rationale:** I originally noted 5-minute in PLAN/DATA. On inspection,
ERCOT RTM SPPs (the settlement prices) are published per 15-minute
settlement interval. 5-minute data is SCED LMPs (dispatch signals), which
are not what operators are paid on. Revenue modeling must use the 15-min
settlement price.

**Revisit when:** we start modeling finer-grained dispatch behavior (e.g.
sub-15-min control responses, ancillary services response time), in which
case we may want to overlay SCED 5-min LMPs.
