# In-progress work

**Status as of 2026-04-26: nothing in flight.** All planned backfills and
experiments completed. Test set revealed; results in [RESULTS.md](RESULTS.md).

This file is kept as a placeholder for future paused work. If a long-running
backfill or experiment is interrupted, document its resume command here so a
future session can pick it up cleanly.

## Past resumption notes (archived)

- 2026-04-25: ERCOT Public API wind+solar forecast backfill paused at
  226/1461 wind docs due to 429 rate limits at 0.3s pacing. Resumed by user
  with `--pause 3.0` to 1,461/1,461 each. Both fully cached at
  `data/raw/ercot_forecasts/{wind,solar}/YYYYMMDD.parquet`.
