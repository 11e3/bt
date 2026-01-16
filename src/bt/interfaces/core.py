"""Core interfaces for the backtesting framework."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from bt.domain.types import Price, Quantity


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Get list of available symbols."""
        ...

    @abstractmethod
    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load data for a symbol."""
        ...

    @abstractmethod
    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get a single bar of data."""
        ...

    @abstractmethod
    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars of data."""
        ...

    @abstractmethod
    def has_more_data(self) -> bool:
        """Check if there's more data to process."""
        ...

    @abstractmethod
    def next_bar(self) -> None:
        """Move to next bar."""
        ...

    @abstractmethod
    def set_current_bar(self, symbol: str, index: int) -> None:
        """Set current bar index."""
        ...

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get current close prices for multiple symbols. Default implementation."""
        prices = {}
        for symbol in symbols:
            bar = self.get_bar(symbol)
            if bar is not None:
                prices[symbol] = float(bar["close"])
        return prices

    def get_current_datetime_batch(self, symbols: list[str]) -> datetime | None:
        """Get current datetime for symbols. Default implementation."""
        for symbol in symbols:
            bar = self.get_bar(symbol)
            if bar is not None:
                dt = bar["datetime"]
                if isinstance(dt, datetime):
                    return dt
                if hasattr(dt, "to_pydatetime"):
                    return dt.to_pydatetime()
                return pd.to_datetime(dt).to_pydatetime()
        return None


class Portfolio(ABC):
    """Abstract base class for portfolio management."""

    @property
    @abstractmethod
    def cash(self) -> Decimal:
        """Available cash."""
        ...

    @property
    @abstractmethod
    def value(self) -> Decimal:
        """Total portfolio value."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Any:
        """Get position for symbol."""
        ...

    @abstractmethod
    def buy(
        self,
        symbol: str,
        price: Price,
        quantity: Quantity,
        date: datetime,
    ) -> bool:
        """Buy shares.

        Returns:
            True if order executed successfully, False otherwise
        """
        ...

    @abstractmethod
    def sell(
        self,
        symbol: str,
        price: Price,
        quantity: Quantity,
        date: datetime,
    ) -> bool:
        """Sell shares.

        Returns:
            True if order executed successfully, False otherwise
        """
        ...

    @abstractmethod
    def update_equity(self, date: datetime, prices: dict[str, Decimal]) -> None:
        """Update portfolio equity."""
        ...

    @property
    @abstractmethod
    def trades(self) -> list[Any]:
        """Get all trades."""
        ...


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def on_bar(self) -> None:
        """Called on each bar."""
        ...


class BacktestEngine(ABC):
    """Abstract base class for backtest engines."""

    @abstractmethod
    def run(self) -> None:
        """Run the backtest."""
        ...


class PerformanceMetrics(ABC):
    """Abstract base class for performance metrics."""

    @abstractmethod
    def calculate(self) -> dict[str, Any]:
        """Calculate performance metrics."""
        ...


class Configuration(ABC):
    """Abstract base class for configuration."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...


class Plugin(ABC):
    """Abstract base class for plugins."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize plugin."""
        ...


# Custom exceptions


class BacktestError(Exception):
    """Base exception for backtesting errors."""

    pass


class DataError(BacktestError):
    """Data-related errors."""

    pass


class ValidationError(BacktestError):
    """Validation errors."""

    pass


class ConfigurationError(BacktestError):
    """Configuration errors."""

    pass


class InsufficientDataError(DataError):
    """Insufficient data errors."""

    pass
