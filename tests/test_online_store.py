"""Tests for online store push and get(zone_id)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meter.config import Settings
from meter.features.aggregate import trips_to_zone_hours
from meter.features.compute import compute_zone_features
from meter.ingest import generate_sample_trips
from meter.online.store import (
    LocalOnlineStore,
    get_features,
    latest_features_by_zone,
    push_latest_to_online,
)


def _tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        raw={
            "project": {"name": "meter", "entity": "zone_id", "feature_version": "v1"},
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "offline_dir": str(tmp_path / "offline"),
                "online_dir": str(tmp_path / "online"),
                "training_dir": str(tmp_path / "training"),
                "sample_dir": str(tmp_path / "sample"),
            },
            "training": {
                "label_horizon_hours": 1,
                "high_demand_threshold_trips": 50,
            },
            "ingest": {"sample_rows": 1000},
            "online": {"backend": "local"},
        }
    )


def test_latest_features_by_zone_keeps_newest():
    features = pd.DataFrame(
        {
            "zone_id": [1, 1, 2],
            "as_of_ts": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 11:00"]
            ),
            "trips_1h": [1.0, 9.0, 3.0],
            "feature_version": ["v1", "v1", "v1"],
        }
    )
    latest = latest_features_by_zone(features)
    assert len(latest) == 2
    z1 = latest.loc[latest["zone_id"] == 1].iloc[0]
    assert z1["trips_1h"] == 9.0


def test_local_store_put_and_get(tmp_path: Path):
    store = LocalOnlineStore(tmp_path / "zone_features.json")
    rows = pd.DataFrame(
        {
            "zone_id": [42],
            "as_of_ts": [pd.Timestamp("2024-01-01 12:00")],
            "feature_version": ["v1"],
            "trips_1h": [12.0],
            "trips_24h": [100.0],
            "trips_7d": [700.0],
            "avg_fare_24h": [15.5],
            "avg_distance_24h": [2.1],
            "computed_at": [pd.Timestamp("2024-01-01 12:05")],
        }
    )
    assert store.put_many(rows) == 1
    got = store.get(42)
    assert got is not None
    assert got["zone_id"] == 42
    assert got["trips_1h"] == 12.0
    assert store.get(999) is None


def test_push_latest_then_get(tmp_path: Path):
    settings = _tmp_settings(tmp_path)
    settings.ensure_dirs()

    trips = generate_sample_trips(n_rows=2_000, days=3, n_zones=5, seed=7)
    features = compute_zone_features(trips_to_zone_hours(trips))
    feat_path = settings.path("offline_dir") / "zone_features.parquet"
    features.to_parquet(feat_path, index=False)

    online_path = push_latest_to_online(features_path=feat_path, settings=settings)
    assert online_path.exists()

    zone_id = int(features["zone_id"].iloc[0])
    got = get_features(zone_id, settings=settings)
    assert got is not None
    assert got["zone_id"] == zone_id
    assert "trips_1h" in got
    assert "as_of_ts" in got
