"""Unit tests for TBBO + MBP-1 merge logic."""

from __future__ import annotations

from math import isnan

import pandas as pd
import pyarrow as pa
import pytest

from src.data.merge import merge_tbbo_mbp1
from src.data.schemas import L1_SCHEMA


def _make_tbbo_df(rows: list[dict]) -> pd.DataFrame:
    """Create a synthetic TBBO DataFrame matching Databento's to_df() output."""
    defaults = {
        "bid_px_00": 5000.0,
        "bid_sz_00": 10,
        "ask_px_00": 5000.25,
        "ask_sz_00": 12,
        "price": 5000.25,
        "size": 1,
        "side": "A",
    }
    records = []
    for r in rows:
        row = {**defaults, **r}
        records.append(row)

    df = pd.DataFrame(records)
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    return df


def _make_mbp1_df(rows: list[dict]) -> pd.DataFrame:
    """Create a synthetic MBP-1 DataFrame matching Databento's to_df() output."""
    defaults = {
        "bid_px_00": 5000.0,
        "bid_sz_00": 10,
        "ask_px_00": 5000.25,
        "ask_sz_00": 12,
        "price": 0.0,
        "size": 0,
        "side": "",
    }
    records = []
    for r in rows:
        row = {**defaults, **r}
        records.append(row)

    df = pd.DataFrame(records)
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    return df


class TestMergeTbboMbp1:
    def test_tbbo_only(self):
        """TBBO records should become trade events with BBO context."""
        tbbo = _make_tbbo_df([
            {"ts_event": 1_000_000_000, "price": 5000.25, "side": "A"},
        ])
        result = merge_tbbo_mbp1(tbbo, None)

        assert result.num_rows == 1
        assert result.column("event_type")[0].as_py() == "trade"
        assert result.column("trade_price")[0].as_py() == 5000.25
        assert result.column("trade_side")[0].as_py() == 1  # buyer

    def test_mbp1_only(self):
        """MBP-1 records without trades should become quote events."""
        mbp1 = _make_mbp1_df([
            {"ts_event": 1_000_000_000, "bid_px_00": 5000.0, "ask_px_00": 5000.25},
        ])
        result = merge_tbbo_mbp1(None, mbp1)

        assert result.num_rows == 1
        assert result.column("event_type")[0].as_py() == "quote"
        assert isnan(result.column("trade_price")[0].as_py())
        assert result.column("trade_size")[0].as_py() == 0

    def test_merge_dedup_same_timestamp(self):
        """When TBBO and MBP-1 share a timestamp, keep the TBBO record."""
        ts = 1_000_000_000
        tbbo = _make_tbbo_df([{"ts_event": ts, "price": 5000.25}])
        mbp1 = _make_mbp1_df([{"ts_event": ts}])

        result = merge_tbbo_mbp1(tbbo, mbp1)

        assert result.num_rows == 1
        assert result.column("event_type")[0].as_py() == "trade"

    def test_merge_interleaved(self):
        """Records should be sorted by timestamp after merge."""
        tbbo = _make_tbbo_df([
            {"ts_event": 2_000_000_000, "price": 5000.50},
        ])
        mbp1 = _make_mbp1_df([
            {"ts_event": 1_000_000_000},
            {"ts_event": 3_000_000_000},
        ])

        result = merge_tbbo_mbp1(tbbo, mbp1)

        assert result.num_rows == 3
        timestamps = result.column("timestamp").to_pylist()
        assert timestamps == sorted(timestamps)
        assert result.column("event_type")[0].as_py() == "quote"
        assert result.column("event_type")[1].as_py() == "trade"
        assert result.column("event_type")[2].as_py() == "quote"

    def test_seller_aggressor(self):
        """Side 'B' (bid) should map to seller aggressor (-1)."""
        tbbo = _make_tbbo_df([
            {"ts_event": 1_000_000_000, "price": 5000.0, "side": "B"},
        ])
        result = merge_tbbo_mbp1(tbbo, None)

        assert result.column("trade_side")[0].as_py() == -1  # seller

    def test_unknown_side(self):
        """Unknown side should map to 0."""
        tbbo = _make_tbbo_df([
            {"ts_event": 1_000_000_000, "price": 5000.0, "side": "N"},
        ])
        result = merge_tbbo_mbp1(tbbo, None)

        assert result.column("trade_side")[0].as_py() == 0

    def test_empty_inputs(self):
        """Both None inputs should return empty table with correct schema."""
        result = merge_tbbo_mbp1(None, None)

        assert result.num_rows == 0
        assert result.schema == L1_SCHEMA

    def test_schema_matches(self):
        """Output schema should match the canonical L1 schema."""
        tbbo = _make_tbbo_df([{"ts_event": 1_000_000_000}])
        result = merge_tbbo_mbp1(tbbo, None)

        assert result.schema == L1_SCHEMA

    def test_multiple_trades_and_quotes(self):
        """Multiple trades and quotes should merge correctly."""
        tbbo = _make_tbbo_df([
            {"ts_event": 1_000_000_000, "price": 5000.25, "side": "A"},
            {"ts_event": 3_000_000_000, "price": 5000.00, "side": "B"},
            {"ts_event": 5_000_000_000, "price": 5000.50, "side": "A"},
        ])
        mbp1 = _make_mbp1_df([
            {"ts_event": 2_000_000_000},
            {"ts_event": 4_000_000_000},
        ])

        result = merge_tbbo_mbp1(tbbo, mbp1)

        assert result.num_rows == 5
        types = result.column("event_type").to_pylist()
        assert types == ["trade", "quote", "trade", "quote", "trade"]
