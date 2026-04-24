"""Project paths. Single source of truth — no hardcoded paths in modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
