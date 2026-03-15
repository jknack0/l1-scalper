"""Unit tests for data validation checks."""

from __future__ import annotations

import tempfile
from datetime import datetime
from math import nan
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.schemas import L1_SCHEMA
from src.data.validation import validate_file

ET = ZoneInfo("America/New_York")


def _ns(hour: int, minute: int, second: int = 0, date_str: str = "2025-10-15") -> int:
    """Create a nanosecond timestamp for a given ET time on a given date."""
    dt = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=ET)
    return int(dt.timestamp() * 1e9)


def _write_table(records: list[dict]) -> str:
    """Write records to a temp Parquet file and return the path."""
    defaults = {
        "bid_price": 5000.0,
        "bid_size": 10,
        "ask_price": 5000.25,
        "ask_size": 12,
        "trade_price": nan,
        "trade_size": 0,
        "trade_side": 0,
        "event_type": "quote",
    }
    rows = [{**defaults, **r} for r in records]
    columns = {field.name: [r[field.name] for r in rows] for field in L1_SCHEMA}
    table = pa.table(columns, schema=L1_SCHEMA)

    f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    pq.write_table(table, f.name)
    return f.name


class TestValidation:
    def test_valid_file(self):
        """Clean data should pass validation."""
        path = _write_table([
            {"timestamp": _ns(10, 0, 0)},
            {"timestamp": _ns(10, 0, 1)},
            {"timestamp": _ns(10, 0, 2)},
        ])
        result = validate_file(path)
        assert result.ok
        assert len(result.warnings) == 0

    def test_non_monotonic(self):
        """Non-monotonic timestamps should be an error."""
        path = _write_table([
            {"timestamp": _ns(10, 0, 2)},
            {"timestamp": _ns(10, 0, 1)},  # goes backwards
            {"timestamp": _ns(10, 0, 3)},
        ])
        result = validate_file(path)
        assert not result.ok
        assert any("Non-monotonic" in e for e in result.errors)

    def test_gap_warning(self):
        """5+ second gap during RTH should warn."""
        path = _write_table([
            {"timestamp": _ns(10, 0, 0)},
            {"timestamp": _ns(10, 0, 6)},  # 6 second gap
        ])
        result = validate_file(path)
        assert result.ok  # warnings don't fail
        assert any("Gap" in w for w in result.warnings)

    def test_gap_error(self):
        """30+ second gap during RTH should error."""
        path = _write_table([
            {"timestamp": _ns(10, 0, 0)},
            {"timestamp": _ns(10, 0, 35)},  # 35 second gap
        ])
        result = validate_file(path)
        assert not result.ok
        assert any("Gap" in e for e in result.errors)

    def test_gap_outside_rth_ok(self):
        """Gaps outside RTH should not trigger warnings."""
        path = _write_table([
            {"timestamp": _ns(8, 0, 0)},   # pre-market
            {"timestamp": _ns(8, 5, 0)},   # 5 min gap, but outside RTH
        ])
        result = validate_file(path)
        assert result.ok

    def test_crossed_quotes(self):
        """Bid >= ask should be an error."""
        path = _write_table([
            {"timestamp": _ns(10, 0, 0), "bid_price": 5001.0, "ask_price": 5000.0},
        ])
        result = validate_file(path)
        assert not result.ok
        assert any("Crossed" in e for e in result.errors)

    def test_trade_through(self):
        """Trade price outside bid/ask should warn."""
        path = _write_table([
            {
                "timestamp": _ns(10, 0, 0),
                "bid_price": 5000.0,
                "ask_price": 5000.25,
                "trade_price": 5001.0,  # way above ask
                "trade_size": 1,
                "trade_side": 1,
                "event_type": "trade",
            },
        ])
        result = validate_file(path)
        assert result.ok  # trade-through is a warning, not error
        assert any("Trade-through" in w for w in result.warnings)

    def test_empty_file(self):
        """Empty file should warn but not error."""
        columns = {field.name: [] for field in L1_SCHEMA}
        table = pa.table(columns, schema=L1_SCHEMA)
        f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        pq.write_table(table, f.name)

        result = validate_file(f.name)
        assert result.ok
        assert any("Empty" in w for w in result.warnings)
