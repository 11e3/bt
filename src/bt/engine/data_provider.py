"""Data provider implementation for backtesting engine."""

from datetime import datetime
from typing import Any

import pandas as pd

from bt.interfaces.core import DataProvider
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryDataProvider(DataProvider):
    """In-memory data provider implementation."""

    def __init__(self) -> None:
        """Initialize data provider."""
        self._data: dict[str, pd.DataFrame] = {}
        self._current_bar: dict[str, int] = {}
        self._cache: dict[str, Any] = {}

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load data for a symbol."""
        # Sort by datetime to ensure chronological order
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)
        self._data[symbol] = df
        self._current_bar[symbol] = 0

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get current bar data for a symbol."""
        if symbol not in self._data:
            return None

        idx = self._current_bar[symbol] + offset
        if idx < 0 or idx >= len(self._data[symbol]):
            return None

        return self._data[symbol].iloc[idx]

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars for a symbol."""
        if symbol not in self._data:
            return None

        start_idx = max(0, self._current_bar[symbol] - count + 1)
        end_idx = self._current_bar[symbol] + 1
        return self._data[symbol].iloc[start_idx:end_idx]

    def get_current_datetime(self, symbol: str) -> datetime | None:
        """Get current datetime for a symbol."""
        if symbol not in self._data or self._current_bar[symbol] >= len(self._data[symbol]):
            return None

        return pd.to_datetime(
            self._data[symbol].iloc[self._current_bar[symbol]]["datetime"]
        ).to_pydatetime()  # type: ignore[no-any-return]

    def next_bar(self) -> None:
        """Move to next bar for all symbols."""
        for symbol in self._data:
            self._current_bar[symbol] += 1

    def has_more_data(self) -> bool:
        """Check if there is more data available after current bar."""
        return any(
            self._current_bar.get(symbol, 0) + 1 < len(self._data.get(symbol, pd.DataFrame()))
            for symbol in self._data
        )

    def set_current_bar(self, symbol: str, index: int) -> None:
        """Set current bar index."""
        if symbol not in self._data:
            raise ValueError(f"Symbol '{symbol}' not loaded")
        if index < 0 or index >= len(self._data[symbol]):
            raise ValueError(f"Index {index} out of bounds for symbol '{symbol}'")
        self._current_bar[symbol] = index

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get current close prices for multiple symbols in one operation.

        Args:
            symbols: List of symbols to get prices for

        Returns:
            Dictionary mapping symbols to their current close prices
        """
        prices = {}
        for symbol in symbols:
            if symbol in self._data and self._current_bar[symbol] < len(self._data[symbol]):
                close_price = self._data[symbol].iloc[self._current_bar[symbol]]["close"]
                prices[symbol] = float(close_price)
        return prices

    def get_current_datetime_batch(self, symbols: list[str]) -> datetime | None:
        """Get current datetime for symbols (returns first non-None datetime).

        Args:
            symbols: List of symbols to check

        Returns:
            Current datetime or None if no data available
        """
        for symbol in symbols:
            if symbol in self._data and self._current_bar[symbol] < len(self._data[symbol]):
                dt = self._data[symbol].iloc[self._current_bar[symbol]]["datetime"]
                if isinstance(dt, datetime):
                    return dt
                if hasattr(dt, "to_pydatetime"):
                    return dt.to_pydatetime()
                return pd.to_datetime(dt).to_pydatetime()
        return None

    @property
    def symbols(self) -> list[str]:
        """Get list of loaded symbols."""
        return list(self._data.keys())
