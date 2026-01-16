"""Protocol interfaces for core backtesting components.

Provides abstract interfaces to break circular dependencies
and enable loose coupling between components.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

if TYPE_CHECKING:
    from bt.domain.types import Amount, Price, Quantity


class IDataProvider(Protocol):
    """Protocol for market data providers."""

    @property
    def symbols(self) -> list[str]:
        """Get list of available symbols."""
        ...

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load data for a symbol."""
        ...

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get a single bar of data."""
        ...

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars of data."""
        ...

    def has_more_data(self) -> bool:
        """Check if there's more data to process."""
        ...

    def next_bar(self) -> None:
        """Move to next bar."""
        ...

    def set_current_bar(self, symbol: str, index: int) -> None:
        """Set current bar index."""
        ...

    def set_current_bars_to_start(self) -> None:
        """Reset all symbols to the start of their data."""
        ...

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices for multiple symbols."""
        ...

    def get_current_datetime_batch(self, symbols: list[str]) -> datetime | None:
        """Get current datetime for symbols."""
        ...


class IPortfolio(Protocol):
    """Protocol for portfolio management."""

    @property
    def cash(self) -> Decimal:
        """Available cash."""
        ...

    @property
    def value(self) -> Decimal:
        """Total portfolio value."""
        ...

    @property
    def trades(self) -> list[Any]:
        """Get all completed trades."""
        ...

    def get_position(self, symbol: str) -> Any:
        """Get position for symbol."""
        ...

    def buy(
        self,
        symbol: str,
        price: "Price",
        quantity: "Quantity",
        date: datetime,
    ) -> bool:
        """Buy shares."""
        ...

    def sell(
        self,
        symbol: str,
        price: "Price",
        quantity: "Quantity",
        date: datetime,
    ) -> bool:
        """Sell shares."""
        ...

    def update_equity(self, date: datetime, prices: dict[str, "Price"]) -> None:
        """Update portfolio equity."""
        ...


class IBacktestEngine(Protocol):
    """Protocol for backtest engines."""

    @property
    def config(self) -> Any:
        """Backtest configuration."""
        ...

    @property
    def data_provider(self) -> IDataProvider:
        """Data provider instance."""
        ...

    @property
    def portfolio(self) -> IPortfolio:
        """Portfolio instance."""
        ...

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load market data for symbol."""
        ...

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get bar data."""
        ...

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars."""
        ...


class IStrategy(Protocol):
    """Protocol for trading strategies."""

    def get_buy_conditions(self) -> dict[str, Any]:
        """Get buy condition functions."""
        ...

    def get_sell_conditions(self) -> dict[str, Any]:
        """Get sell condition functions."""
        ...

    def get_buy_price_func(self) -> Any:
        """Get buy price function."""
        ...

    def get_sell_price_func(self) -> Any:
        """Get sell price function."""
        ...

    def get_allocation_func(self) -> Any:
        """Get allocation function."""
        ...


class ILogger(Protocol):
    """Protocol for logging."""

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...


class IMetricsGenerator(Protocol):
    """Protocol for performance metrics generation."""

    def calculate_metrics(
        self,
        equity_curve: list[Decimal],
        dates: list[datetime],
        trades: list[Any],
        initial_cash: "Amount",
    ) -> Any:
        """Calculate performance metrics."""
        ...


class IChartGenerator(Protocol):
    """Protocol for chart generation."""

    def generate_equity_chart(self, data: Any) -> Any:
        """Generate equity curve chart."""
        ...

    def generate_performance_chart(self, data: Any) -> Any:
        """Generate performance chart."""
        ...


# Type aliases for better readability
ConditionFunc = Any  # Will be defined properly after circular deps are fixed
PriceFunc = Any
AllocationFunc = Any
