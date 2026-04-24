"""Smoke tests for the ERCOT fetcher that don't hit the network.

We mock gridstatus.Ercot so tests are offline. The focus here is the caching
layer and schema normalization, not the gridstatus API itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _fake_raw_df(year: int = 2023) -> pd.DataFrame:
    """Shape of what gridstatus.Ercot.get_rtm_spp returns, parameterized by year."""
    idx = pd.date_range(f"{year}-01-01", periods=4, freq="15min", tz="US/Central")
    rows = []
    for loc in ["HB_NORTH", "HB_HOUSTON"]:
        for i, ts in enumerate(idx):
            rows.append(
                {
                    "Time": ts,
                    "Interval Start": ts,
                    "Interval End": ts + pd.Timedelta("15min"),
                    "Location": loc,
                    "Location Type": "Trading Hub",
                    "Market": "REAL_TIME_15_MIN",
                    "SPP": 10.0 + i + (0 if loc == "HB_NORTH" else 100),
                }
            )
    return pd.DataFrame(rows)


def test_year_fetcher_normalizes_schema_and_caches(tmp_path, monkeypatch):
    # Redirect the cache dir for the test.
    from src.data import ercot
    monkeypatch.setattr(ercot, "CACHE_DIR", tmp_path)

    fake_iso = MagicMock()
    fake_iso.get_rtm_spp.return_value = _fake_raw_df()
    with patch.object(ercot, "__import__", create=True):
        with patch("gridstatus.Ercot", return_value=fake_iso):
            df = ercot.get_rtm_spp_year(2023)

    # Schema checks
    assert set(df.columns) == {"timestamp_utc", "location", "spp"}
    assert str(df["timestamp_utc"].dtype).startswith("datetime64[ns, UTC")
    assert (df["location"].isin(["HB_NORTH", "HB_HOUSTON"])).all()
    assert (df["spp"] >= 0).all()

    # Cache file created
    cached = tmp_path / "rtm_spp_2023.parquet"
    assert cached.exists()

    # Second call reads from cache without calling gridstatus.
    fake_iso.get_rtm_spp.reset_mock()
    with patch("gridstatus.Ercot", return_value=fake_iso):
        df2 = ercot.get_rtm_spp_year(2023)
    fake_iso.get_rtm_spp.assert_not_called()
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_get_series_filters_and_concats(tmp_path, monkeypatch):
    from src.data import ercot
    monkeypatch.setattr(ercot, "CACHE_DIR", tmp_path)

    fake_iso = MagicMock()
    fake_iso.get_rtm_spp.side_effect = [_fake_raw_df(2022), _fake_raw_df(2023)]
    with patch("gridstatus.Ercot", return_value=fake_iso):
        s = ercot.get_rtm_spp_series("HB_NORTH", 2022, 2023)

    assert isinstance(s, pd.Series)
    assert s.name == "rtm_spp_HB_NORTH"
    assert str(s.index.dtype).startswith("datetime64[ns, UTC")
    assert len(s) == 8
    # Monotonically increasing UTC index after concat+sort.
    assert s.index.is_monotonic_increasing
    # Only HB_NORTH rows (values start at 10, HB_HOUSTON would be 110+).
    assert s.max() < 20


def test_year_before_archive_raises(tmp_path, monkeypatch):
    from src.data import ercot
    monkeypatch.setattr(ercot, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError):
        ercot.get_rtm_spp_year(2010)
