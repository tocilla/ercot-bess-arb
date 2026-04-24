"""Backfill ERCOT Public API wind and solar forecast archives.

One doc per day per endpoint, published around 06:00 UTC (early-morning
CT, day-ahead vintage). Sequential with sleep pauses to respect rate
limits.

Usage:
    python scripts/backfill_ercot_forecasts.py \
        --start 2019-01-01 --end 2022-12-31 \
        --endpoints wind,solar
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

from src.data.ercot_forecasts import backfill_daily_forecasts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--endpoints", default="wind,solar",
                    help="comma-sep endpoint keys")
    ap.add_argument("--publish-hour-utc", type=int, default=6)
    ap.add_argument("--pause", type=float, default=0.5,
                    help="seconds to pause between requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    keys = [k.strip() for k in args.endpoints.split(",")]

    for key in keys:
        print(f"\n=== {key} {start.date()} → {end.date()} ===")
        t0 = time.time()
        last_report = [t0]

        def on_progress(done: int, total: int):
            now = time.time()
            if now - last_report[0] >= 10.0 or done == total:
                elapsed = now - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(f"  [{key}] progress: {done}/{total} "
                      f"({done / total * 100:.1f}%) "
                      f"elapsed={elapsed:.0f}s rate={rate:.2f}/s "
                      f"eta={eta:.0f}s")
                last_report[0] = now

        backfill_daily_forecasts(
            key, start, end,
            target_publish_hour_utc=args.publish_hour_utc,
            pause_seconds=args.pause,
            on_progress=on_progress,
        )
        print(f"  [{key}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
