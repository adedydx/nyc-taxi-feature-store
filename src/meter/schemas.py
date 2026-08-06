"""Canonical schemas for Meter entities, events, features, and training rows."""

from __future__ import annotations

TRIP_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
    "pulocation_id",
    "dolocation_id",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "payment_type",
]

ZONE_HOUR_COLUMNS = [
    "zone_id",
    "event_hour",
    "trip_count",
    "total_fare",
    "avg_fare",
    "avg_distance",
    "avg_passengers",
]

FEATURE_COLUMNS = [
    "zone_id",
    "as_of_ts",
    "feature_version",
    "trips_1h",
    "trips_24h",
    "trips_7d",
    "avg_fare_24h",
    "avg_distance_24h",
    "computed_at",
]

TRAINING_COLUMNS = FEATURE_COLUMNS + [
    "prediction_ts",
    "label_trips_next_1h",
    "label_high_demand",
]

ONLINE_KEY = "zone_id"
ONLINE_PAYLOAD_FIELDS = [
    "as_of_ts",
    "feature_version",
    "trips_1h",
    "trips_24h",
    "trips_7d",
    "avg_fare_24h",
    "avg_distance_24h",
    "computed_at",
]
