"""Generate TLC-style sample trip data for local demos (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meter.config import Settings, get_settings
from meter.schemas import TRIP_COLUMNS


def generate_sample_trips(
    n_rows: int = 50_000,
    start: str = "2024-01-01",
    days: int = 14,
    n_zones: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    zone_weights = rng.dirichlet(np.ones(n_zones) * 0.4)
    zones = np.arange(1, n_zones + 1)

    offsets = rng.integers(0, days * 24 * 3600, size=n_rows)
    pickup = start_ts + pd.to_timedelta(offsets, unit="s")
    duration_min = rng.integers(3, 45, size=n_rows)
    dropoff = pickup + pd.to_timedelta(duration_min, unit="m")

    pulocation = rng.choice(zones, size=n_rows, p=zone_weights)
    dolocation = rng.choice(zones, size=n_rows)
    distance = np.round(rng.uniform(0.5, 12.0, size=n_rows), 2)
    fare = np.round(2.5 + distance * rng.uniform(1.8, 3.2, size=n_rows), 2)

    df = pd.DataFrame(
        {
            "pickup_datetime": pickup,
            "dropoff_datetime": dropoff,
            "pulocation_id": pulocation.astype(int),
            "dolocation_id": dolocation.astype(int),
            "passenger_count": rng.integers(1, 5, size=n_rows),
            "trip_distance": distance,
            "fare_amount": fare,
            "total_amount": np.round(fare * 1.15, 2),
            "payment_type": rng.choice([1, 2], size=n_rows, p=[0.7, 0.3]),
        }
    )
    return df[TRIP_COLUMNS].sort_values("pickup_datetime").reset_index(drop=True)


def write_sample_trips(
    settings: Settings | None = None,
    n_rows: int | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.ensure_dirs()
    n = n_rows or int(settings.raw["ingest"]["sample_rows"])
    df = generate_sample_trips(n_rows=n)
    out = settings.path("raw_dir") / "yellow_trips.parquet"
    df.to_parquet(out, index=False)
    return out
