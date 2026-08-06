"""Local online store: latest zone features keyed by zone_id (DynamoDB stand-in)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from meter.config import Settings, get_settings
from meter.schemas import ONLINE_KEY, ONLINE_PAYLOAD_FIELDS


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def latest_features_by_zone(features: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest feature row per zone_id (max as_of_ts)."""
    df = features.copy()
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"])
    df = df.sort_values(["zone_id", "as_of_ts"])
    return df.groupby("zone_id", as_index=False).tail(1).reset_index(drop=True)


class LocalOnlineStore:
    """JSON file keyed by zone_id. Swap later for DynamoDB without changing callers."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open() as f:
            return json.load(f)

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            json.dump(data, f, indent=2, default=_json_default)
            f.write("\n")

    def put_many(self, rows: pd.DataFrame) -> int:
        data = self._load()
        for record in rows.to_dict(orient="records"):
            zone_id = int(record[ONLINE_KEY])
            payload = {k: record[k] for k in ONLINE_PAYLOAD_FIELDS if k in record}
            payload[ONLINE_KEY] = zone_id
            data[str(zone_id)] = payload
        self._save(data)
        return len(rows)

    def get(self, zone_id: int) -> dict[str, Any] | None:
        data = self._load()
        return data.get(str(int(zone_id)))

    def size(self) -> int:
        return len(self._load())


def _store_path(settings: Settings) -> Path:
    return settings.path("online_dir") / "zone_features.json"


def push_latest_to_online(
    features_path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Materialize latest-per-zone from offline features into the online store."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    path = features_path or (settings.path("offline_dir") / "zone_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Offline features not found at {path}. Run `meter offline` first."
        )

    features = pd.read_parquet(path)
    latest = latest_features_by_zone(features)
    store = LocalOnlineStore(_store_path(settings))
    store.put_many(latest)
    return store.path


def get_features(
    zone_id: int,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """Low-latency lookup of the latest features for a zone."""
    settings = settings or get_settings()
    return LocalOnlineStore(_store_path(settings)).get(zone_id)
