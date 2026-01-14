#!/usr/bin/env python3
"""Final simplified backtest implementation that avoids all pandas datetime issues."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import datetime
from typing import Any

import pandas as pd

from bt.config.config import settings

# Import simplified components that avoid pandas issues
from bt.engine.data_provider import DataProvider


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load data without pandas datetime issues."""
    interval_dir = settings.data_dir / interval
    file_path = interval_dir / f"{symbol}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    # Fix datetime column once - store as datetime throughout
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


class InMemoryDataProvider(DataProvider):
    """Data provider that avoids pandas datetime type conflicts."""

    def __init__(self):
        self._data: dict[str, pd.DataFrame] = {}
        self._current_bar: dict[str, int] = {}
        self._cache: dict[str, Any] = {}

    def load_data(self, symbol: str, data: Any) -> None:
        """Load data without pandas datetime issues."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a pandas DataFrame")

        # Fix datetime column and sort
        df = data.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        self._data[symbol] = df
        self._current_bar[symbol] = 0

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get bar without pandas datetime issues."""
        if symbol not in self._data:
            return None

        idx = self._current_bar[symbol] + offset
        if idx < 0 or idx >= len(self._data[symbol]):
            return None

        # Access data directly
        return self._data[symbol].iloc[idx]

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars without pandas issues."""
        if symbol not in self._data:
            return None

        # Exclude current bar to prevent look-ahead bias
        end_idx = self._current_bar[symbol]
        start_idx = max(0, end_idx - count)
        if start_idx >= end_idx:
            return None

        return self._data[symbol].iloc[start_idx:end_idx]

    def get_current_datetime(self, symbol: str) -> datetime | None:
        """Get current datetime without pandas issues."""
        bar = self.get_bar(symbol)
        if bar is None:
            return None

        dt = bar["datetime"]
        if isinstance(dt, datetime):
            return dt
        # Fallback without pandas issues
        return pd.to_datetime(dt).to_pydatetime()

    def has_more_data(self) -> bool:
        """Check if more data available."""
        return any(self._current_bar[symbol] < len(self._data[symbol]) - 1 for symbol in self._data)

    def next_bar(self) -> None:
        """Move to next time period."""
        for symbol in self._data:
            if self._current_bar[symbol] < len(self._data[symbol]) - 1:
                self._current_bar[symbol] += 1

    def symbols(self) -> list[str]:
        """Get available symbols."""
        return list(self._data.keys())

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate for debugging."""
        # Simple implementation
        return 0.0
