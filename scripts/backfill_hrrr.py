"""Parallel HRRR backfill for a date range.

Defaults match the minimal-scope choice from FINDINGS 2026-04-24:
one cycle per day (12Z — late-morning UTC = early-morning CT, available
before ERCOT DAM close at 10am local), one forecast hour (F06 —
forecast valid at 18Z = 12:00-13:00 CT, the morning ramp).

Usage:
    python scripts/backfill_hrrr.py --start 2019-01-01 --end 2022-12-31
    python scripts/backfill_hrrr.py --start 2020-11-01 --end 2022-12-31 \
        --cycles 6,12,18 --fxx 3,6,12
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hrrr import fetch_hrrr_range_parallel  # noqa: E402


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--cycles", default="12", help="comma-sep UTC cycle hours")
    ap.add_argument("--fxx", default="6", help="comma-sep forecast hours")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    cycles = _parse_ints(args.cycles)
    fxx = _parse_ints(args.fxx)

    print(f"Backfilling HRRR {start.date()} → {end.date()}")
    print(f"  cycles: {cycles}  fxx: {fxx}  workers: {args.workers}")

    t0 = time.time()
    last_report = [t0]

    def on_progress(done: int, total: int):
        now = time.time()
        if now - last_report[0] >= 10.0 or done == total:
            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(f"  progress: {done}/{total} ({done/total*100:.1f}%) "
                  f"elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s")
            last_report[0] = now

    df = fetch_hrrr_range_parallel(start, end, cycles=cycles, fxx_range=fxx,
                                   max_workers=args.workers,
                                   on_progress=on_progress)
    total_seconds = time.time() - t0
    print(f"\nDone in {total_seconds:.0f}s ({total_seconds/60:.1f} min). "
          f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
