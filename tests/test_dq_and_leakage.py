"""Tests for DQ gates and leakage demo."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from meter.config import Settings
from meter.dq.checks import (
    check_feature_non_negative,
    check_feature_nulls,
    check_ingest,
    check_materialization_freshness,
    check_online_offline_parity,
    check_volume_drop,
    run_dq_suite,
)
from meter.dq.leakage import run_leakage_demo
from meter.features.aggregate import trips_to_zone_hours
from meter.features.compute import compute_zone_features
from meter.ingest import generate_sample_trips
from meter.online.store import LocalOnlineStore, push_latest_to_online
from meter.training.dataset import build_training_set


def _settings(tmp_path: Path) -> Settings:
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
            "freshness": {
                "zone_trip_features_ttl_hours": 1,
                "online_max_staleness_hours": 2,
                "alert_if_lag_hours": 2,
            },
            "dq": {
                "max_null_rate": 0.01,
                "max_volume_drop_pct": 0.40,
                "parity_sample_size": 20,
                "parity_max_rel_error": 0.05,
            },
            "ingest": {"sample_rows": 2000},
            "online": {"backend": "local"},
        }
    )


def _materialize(settings: Settings) -> None:
    settings.ensure_dirs()
    trips = generate_sample_trips(n_rows=3_000, days=4, n_zones=8, seed=3)
    trips_path = settings.path("raw_dir") / "yellow_trips.parquet"
    trips.to_parquet(trips_path, index=False)
    zh = trips_to_zone_hours(trips)
    zh.to_parquet(settings.path("offline_dir") / "zone_hours.parquet", index=False)
    feats = compute_zone_features(zh)
    feats.to_parquet(settings.path("offline_dir") / "zone_features.parquet", index=False)
    build_training_set(settings=settings)
    push_latest_to_online(settings=settings)


def test_ingest_rejects_empty():
    result = check_ingest(pd.DataFrame())
    assert not result.passed


def test_null_and_negative_checks():
    good = pd.DataFrame(
        {
            "zone_id": [1],
            "as_of_ts": [pd.Timestamp("2024-01-01")],
            "feature_version": ["v1"],
            "trips_1h": [1.0],
            "trips_24h": [2.0],
            "trips_7d": [3.0],
            "avg_fare_24h": [10.0],
            "avg_distance_24h": [2.0],
            "computed_at": [pd.Timestamp("2024-01-01")],
        }
    )
    assert check_feature_nulls(good, 0.01).passed
    assert check_feature_non_negative(good).passed

    bad = good.copy()
    bad.loc[0, "trips_1h"] = -1
    assert not check_feature_non_negative(bad).passed


def test_freshness_fails_when_stale():
    feats = pd.DataFrame(
        {"computed_at": [pd.Timestamp("2020-01-01 00:00:00")]}
    )
    now = datetime(2024, 1, 1)
    result = check_materialization_freshness(feats, max_lag_hours=2, now=now)
    assert not result.passed


def test_freshness_passes_when_recent():
    now = datetime(2024, 1, 1, 12, 0, 0)
    feats = pd.DataFrame(
        {"computed_at": [pd.Timestamp(now - timedelta(minutes=30))]}
    )
    result = check_materialization_freshness(feats, max_lag_hours=2, now=now)
    assert result.passed


def test_volume_drop_detects_collapse():
    hours = pd.date_range("2024-01-01", periods=48, freq="h")
    counts = [100] * 24 + [1] * 24
    zh = pd.DataFrame(
        {
            "zone_id": [1] * 48,
            "event_hour": hours,
            "trip_count": counts,
            "total_fare": counts,
            "avg_fare": [1.0] * 48,
            "avg_distance": [1.0] * 48,
            "avg_passengers": [1.0] * 48,
        }
    )
    result = check_volume_drop(zh, max_drop_pct=0.40)
    assert not result.passed


def test_parity_detects_mismatch(tmp_path: Path):
    offline = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "as_of_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
            "trips_1h": [10.0, 20.0],
        }
    )
    store = LocalOnlineStore(tmp_path / "online.json")
    store.put_many(
        pd.DataFrame(
            {
                "zone_id": [1, 2],
                "as_of_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
                "feature_version": ["v1", "v1"],
                "trips_1h": [10.0, 999.0],  # zone 2 skewed
                "trips_24h": [0.0, 0.0],
                "trips_7d": [0.0, 0.0],
                "avg_fare_24h": [0.0, 0.0],
                "avg_distance_24h": [0.0, 0.0],
                "computed_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:00"]),
            }
        )
    )
    result = check_online_offline_parity(
        offline, store, sample_size=2, max_rel_error=0.05
    )
    assert not result.passed


def test_dq_suite_passes_on_fresh_pipeline(tmp_path: Path):
    settings = _settings(tmp_path)
    _materialize(settings)
    report = run_dq_suite(settings=settings)
    assert report.passed, report.failed_names()


def test_leakage_demo_shows_stronger_leaky_corr(tmp_path: Path):
    settings = _settings(tmp_path)
    _materialize(settings)
    result = run_leakage_demo(settings=settings)
    assert result.leaky_exact_match_rate == pytest.approx(1.0)
    assert result.leaky_label_corr > result.correct_feature_label_corr
