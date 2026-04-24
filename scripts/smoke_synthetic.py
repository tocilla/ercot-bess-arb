"""End-to-end smoke run on synthetic prices.

Generates a month of synthetic LMPs with a daily sine shape, runs the
natural-spread oracle baseline, and prints summary metrics. This verifies
the scaffold works before touching real data.

Usage:
    python scripts/smoke_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import daily_oracle_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.metrics import compare, summarize  # noqa: E402
from src.synthetic import synthetic_lmp  # noqa: E402


def main() -> None:
    spec = BatterySpec(
        power_mw=100.0,
        capacity_mwh=200.0,
        roundtrip_eff=0.85,
        soc_min_frac=0.05,
        soc_max_frac=0.95,
        initial_soc_frac=0.5,
        degradation_cost_per_mwh=2.0,
    )

    prices = synthetic_lmp(
        start="2024-01-01",
        days=30,
        interval_minutes=60,
        mean_price=40.0,
        daily_amplitude=25.0,
        noise_std=3.0,
        seed=0,
    )

    oracle_sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    oracle_result = run_dispatch(oracle_sched, prices, spec, interval_hours=1.0)

    # A "do nothing" run, as a sanity reference.
    import pandas as pd
    idle_sched = pd.Series(0.0, index=prices.index, name="grid_power_mw")
    idle_result = run_dispatch(idle_sched, prices, spec, interval_hours=1.0)

    summary = compare({"idle": idle_result, "natural_spread_oracle": oracle_result})
    print("\n=== Synthetic LMP smoke run ===")
    print(f"Days: {len(prices) // 24}, interval: 60min, price range: "
          f"${prices.min():.2f} – ${prices.max():.2f} (mean ${prices.mean():.2f})")
    print("\nSummary:")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
