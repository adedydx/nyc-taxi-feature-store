"""Zone feature computation with explicit as-of timestamps (no future leakage)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from meter.config import Settings, get_settings
from meter.schemas import FEATURE_COLUMNS


def compute_zone_features(
    zone_hours: pd.DataFrame,
    feature_version: str = "v1",
) -> pd.DataFrame:
    """
    For each (zone_id, event_hour), compute features using only rows at or before that hour.

    Rolling windows run on a continuous hourly grid per zone so gaps don't silently
    pull in older hours as if they were adjacent.
    """
    df = zone_hours.copy()
    df["event_hour"] = pd.to_datetime(df["event_hour"])
    df = df.sort_values(["zone_id", "event_hour"])

    parts: list[pd.DataFrame] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for zone_id, g in df.groupby("zone_id", sort=False):
        g = g.set_index("event_hour").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_idx)
        g["trip_count"] = g["trip_count"].fillna(0)
        g["total_fare"] = g["total_fare"].fillna(0.0)
        g["avg_fare"] = g["avg_fare"].fillna(0.0)
        g["avg_distance"] = g["avg_distance"].fillna(0.0)

        out = pd.DataFrame(
            {
                "zone_id": zone_id,
                "as_of_ts": g.index,
                "feature_version": feature_version,
                "trips_1h": g["trip_count"].rolling(window=1, min_periods=1).sum(),
                "trips_24h": g["trip_count"].rolling(window=24, min_periods=1).sum(),
                "trips_7d": g["trip_count"].rolling(window=24 * 7, min_periods=1).sum(),
                "avg_fare_24h": g["avg_fare"].rolling(window=24, min_periods=1).mean(),
                "avg_distance_24h": g["avg_distance"]
                .rolling(window=24, min_periods=1)
                .mean(),
                "computed_at": now,
            }
        )
        parts.append(out.reset_index(drop=True))

    features = pd.concat(parts, ignore_index=True)
    return features[FEATURE_COLUMNS]


def materialize_features(
    zone_hours_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.ensure_dirs()
    path = zone_hours_path or (settings.path("offline_dir") / "zone_hours.parquet")
    zone_hours = pd.read_parquet(path)
    version = settings.raw["project"]["feature_version"]
    features = compute_zone_features(zone_hours, feature_version=version)
    out = settings.path("offline_dir") / "zone_features.parquet"
    features.to_parquet(out, index=False)
    return out
