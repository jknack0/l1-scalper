from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatabentoSettings:
    api_key: str = field(default_factory=lambda: os.environ["DATABENTO_API_KEY"])
    dataset: str = "GLBX.MDP3"
    symbol: str = "MES.FUT"
    stype: str = "continuous"


@dataclass(frozen=True)
class StorageSettings:
    historical_dir: Path = field(default_factory=lambda: DATA_DIR / "l1" / "historical")
    live_dir: Path = field(default_factory=lambda: DATA_DIR / "l1" / "live")

    def __post_init__(self) -> None:
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)


# RTH session bounds (Eastern)
RTH_START_HOUR = 9
RTH_START_MIN = 30
RTH_END_HOUR = 16
RTH_END_MIN = 0
