from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return ROOT

    def path(self, key: str) -> Path:
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    def ensure_dirs(self) -> None:
        for key in ("raw_dir", "offline_dir", "online_dir", "training_dir", "sample_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        cfg_path = path or DEFAULT_CONFIG
        with cfg_path.open() as f:
            data = yaml.safe_load(f)
        return cls(raw=data)


def get_settings(path: Path | None = None) -> Settings:
    return Settings.load(path)
