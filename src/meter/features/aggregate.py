"""Aggregate raw trips to zone-hour fact table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meter.config import Settings, get_settings
from meter.schemas import ZONE_HOUR_COLUMNS


def trips_to_zone_hours(trips: pd.DataFrame) -> pd.DataFrame:
    df = trips.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["event_hour"] = df["pickup_datetime"].dt.floor("h")
    df = df.rename(columns={"pulocation_id": "zone_id"})

    g = (
        df.groupby(["zone_id", "event_hour"], as_index=False)
        .agg(
            trip_count=("fare_amount", "size"),
            total_fare=("fare_amount", "sum"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_passengers=("passenger_count", "mean"),
        )
        .sort_values(["zone_id", "event_hour"])
        .reset_index(drop=True)
    )
    return g[ZONE_HOUR_COLUMNS]


def materialize_zone_hours(
    trips_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.ensure_dirs()
    path = trips_path or (settings.path("raw_dir") / "yellow_trips.parquet")
    trips = pd.read_parquet(path)
    zone_hours = trips_to_zone_hours(trips)
    out = settings.path("offline_dir") / "zone_hours.parquet"
    zone_hours.to_parquet(out, index=False)
    return out
