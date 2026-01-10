"""Data provider for market data access during backtesting.

Manages market data and provides bar-level access with proper indexing.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from bt.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

    pass

logger = get_logger(__name__)


class DataProvider:
    """Provides market data access for backtesting.

    Manages:
    - Data loading and storage
    - Current bar tracking
    - Historical data retrieval
    - Data synchronization across symbols
    """

    def __init__(self) -> None:
        """Initialize data provider."""
        self._data: dict[str, pd.DataFrame] = {}
        self._current_bar: dict[str, int] = {}

        logger.debug("DataProvider initialized")

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load market data for a symbol.

        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data and datetime column
        """
        # Validate required columns
        required_cols = ["datetime", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort and reset index
        df = df.copy()
        df = df.sort_values("datetime").reset_index(drop=True)

        self._data[symbol] = df
        self._current_bar[symbol] = 0

        logger.info(
            "Data loaded",
            extra={
                "symbol": symbol,
                "rows": len(df),
                "start": df["datetime"].min().isoformat(),
                "end": df["datetime"].max().isoformat(),
            },
        )

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get bar data at current position + offset.

        Args:
            symbol: Trading symbol
            offset: Offset from current bar (0 = current, -1 = previous)

        Returns:
            Bar data as Series, or None if out of bounds
        """
        if symbol not in self._data:
            return None

        idx = self._current_bar[symbol] + offset
        if idx < 0 or idx >= len(self._data[symbol]):
            return None

        return self._data[symbol].iloc[idx]

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars ending at current position.

        Args:
            symbol: Trading symbol
            count: Number of bars to retrieve

        Returns:
            DataFrame with bar data, or None if insufficient data
        """
        if symbol not in self._data:
            return None

        end_idx = self._current_bar[symbol] + 1
        start_idx = max(0, end_idx - count)

        if start_idx >= end_idx:
            return None

        return self._data[symbol].iloc[start_idx:end_idx]

    def has_more_data(self) -> bool:
        """Check if there is more data to process.

        Returns:
            True if any symbol has unprocessed bars
        """
        return any(self._current_bar[symbol] < len(self._data[symbol]) - 1 for symbol in self._data)

    def next_bar(self) -> None:
        """Advance to next bar for all symbols."""
        for symbol in self._data:
            if self._current_bar[symbol] < len(self._data[symbol]) - 1:
                self._current_bar[symbol] += 1

    def set_current_bar(self, symbol: str, index: int) -> None:
        """Set current bar position for a symbol.

        Args:
            symbol: Trading symbol
            index: Bar index to set
        """
        if symbol not in self._data:
            raise ValueError(f"Symbol {symbol} not loaded")

        if index < 0 or index >= len(self._data[symbol]):
            raise ValueError(f"Index {index} out of bounds for {symbol}")

        self._current_bar[symbol] = index

    def get_current_datetime(self, symbol: str) -> datetime | None:
        """Get datetime of current bar.

        Args:
            symbol: Trading symbol

        Returns:
            Datetime of current bar, or None if no data
        """
        bar = self.get_bar(symbol)
        if bar is None:
            return None
        dt = bar["datetime"]
        return dt if isinstance(dt, datetime) else dt.to_pydatetime()

    @property
    def symbols(self) -> list[str]:
        """Get list of loaded symbols."""
        return list(self._data.keys())
