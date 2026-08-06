"""CLI for offline/online paths, DQ gates, and leakage demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _run_offline(sample_rows: int | None = None) -> dict[str, Path]:
    from meter.config import get_settings
    from meter.features.aggregate import materialize_zone_hours
    from meter.features.compute import materialize_features
    from meter.ingest import write_sample_trips
    from meter.training.dataset import build_training_set

    settings = get_settings()
    settings.ensure_dirs()

    trips = write_sample_trips(settings=settings, n_rows=sample_rows)
    zone_hours = materialize_zone_hours(trips_path=trips, settings=settings)
    features = materialize_features(zone_hours_path=zone_hours, settings=settings)
    training = build_training_set(settings=settings)

    return {
        "trips": trips,
        "zone_hours": zone_hours,
        "features": features,
        "training": training,
    }


def _run_online() -> Path:
    from meter.online.store import push_latest_to_online

    return push_latest_to_online()


def _run_get(zone_id: int) -> int:
    from meter.online.store import get_features

    payload = get_features(zone_id)
    if payload is None:
        print(f"No online features for zone_id={zone_id}")
        return 1
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _run_dq() -> int:
    from meter.config import get_settings
    from meter.dq.checks import run_dq_suite, write_dq_report

    settings = get_settings()
    report = run_dq_suite(settings=settings)
    out = write_dq_report(report, settings.path("offline_dir") / "dq_report.json")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"DQ report written: {out}")
    if not report.passed:
        print(f"DQ FAILED: {report.failed_names()}")
        return 1
    print("DQ PASSED")
    return 0


def _run_leakage() -> int:
    from meter.dq.leakage import run_leakage_demo

    result = run_leakage_demo()
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meter",
        description="Meter — NYC taxi zone feature store",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser(
        "offline",
        help="Run sample ingest → zone-hour → features → PIT training set",
    )
    offline.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Override sample trip row count (default from config)",
    )

    sub.add_parser(
        "online",
        help="Push latest-per-zone features from offline store to online store",
    )

    get_cmd = sub.add_parser(
        "get",
        help="Low-latency lookup of latest online features for a zone",
    )
    get_cmd.add_argument("zone_id", type=int, help="Taxi zone id")

    sub.add_parser(
        "dq",
        help="Run DQ gates (ingest, features, freshness, online/offline parity)",
    )
    sub.add_parser(
        "leakage",
        help="Demo leaky join vs correct PIT join on the training set",
    )

    args = parser.parse_args(argv)

    if args.command == "offline":
        paths = _run_offline(sample_rows=args.rows)
        print("Offline path complete:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0

    if args.command == "online":
        path = _run_online()
        print(f"Online push complete: {path}")
        return 0

    if args.command == "get":
        return _run_get(args.zone_id)

    if args.command == "dq":
        return _run_dq()

    if args.command == "leakage":
        return _run_leakage()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
