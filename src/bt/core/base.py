"""Base abstract classes for the backtesting framework."""

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from bt.domain.types import (
    Amount,
    Fee,
    Percentage,
    Price,
    Quantity,
)
from bt.interfaces.core import (
    BacktestEngine,
    Configuration,
    DataProvider,
    PerformanceMetrics,
    Plugin,
    Portfolio,
    Strategy,
)


class BaseDataProvider(DataProvider):
    """Abstract base class for data providers."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._symbols: list[str] = []

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Implement in subclass."""
        raise NotImplementedError

    def get_bar(self, symbol: str, offset: int = 0) -> Any | None:
        """Implement in subclass."""
        raise NotImplementedError

    def get_bars(self, symbol: str, count: int) -> Any | None:
        """Implement in subclass."""
        raise NotImplementedError

    def has_more_data(self) -> bool:
        """Implement in subclass."""
        raise NotImplementedError

    def next_bar(self) -> None:
        """Implement in subclass."""
        raise NotImplementedError

    @property
    def symbols(self) -> list[str]:
        """Get available symbols."""
        return list(self._symbols)


class BasePortfolio(Portfolio):
    """Abstract base class for portfolios."""

    def __init__(self, initial_cash: Amount, fee: Fee, slippage: Percentage):
        self.initial_cash = initial_cash
        self._cash = initial_cash
        self.fee = fee
        self.slippage = slippage
        self._positions: dict[str, Any] = {}
        self._trades: list[Any] = []
        self._equity_curve: list[Decimal] = [Decimal(str(initial_cash))]
        self._dates: list[datetime] = []

    def get_position(self, symbol: str) -> Any:
        """Get position for symbol."""
        return self._positions.get(symbol, self._create_position(symbol))

    def _create_position(self, symbol: str) -> Any:
        """Create position object - implement in subclass."""
        raise NotImplementedError

    def buy(self, symbol: str, price: Price, quantity: Quantity, date: datetime) -> None:
        """Implement in subclass."""
        raise NotImplementedError

    def sell(self, symbol: str, price: Price, quantity: Quantity, date: datetime) -> None:
        """Implement in subclass."""
        raise NotImplementedError

    @property
    def cash(self) -> Amount:
        """Available cash."""
        return self._cash

    @property
    def value(self) -> Amount:
        """Total portfolio value."""
        return self.cash  # Simplified - override in subclass

    @property
    def positions(self) -> dict[str, Any]:
        """All positions."""
        return self._positions

    @property
    def trades(self) -> list[Any]:
        """All trades."""
        return self._trades

    @property
    def equity_curve(self) -> list[Decimal]:
        """Equity curve."""
        return self._equity_curve

    @property
    def dates(self) -> list[datetime]:
        """Equity dates."""
        return self._dates


class BaseStrategy(Strategy):
    """Abstract base class for strategies."""

    def __init__(self, name: str):
        self.name = name

    def get_buy_conditions(self) -> dict[str, Any]:
        """Get buy condition functions."""
        raise NotImplementedError

    def get_sell_conditions(self) -> dict[str, Any]:
        """Get sell condition functions."""
        raise NotImplementedError

    def get_buy_price_func(self) -> Any:
        """Get buy price function."""
        raise NotImplementedError

    def get_sell_price_func(self) -> Any:
        """Get sell price function."""
        raise NotImplementedError

    def get_allocation_func(self) -> Any:
        """Get allocation function."""
        raise NotImplementedError


class BaseBacktestEngine(BacktestEngine):
    """Abstract base class for backtest engines."""

    def __init__(self, config: Configuration, data_provider: DataProvider, portfolio: Portfolio):
        self.config = config
        self.data_provider = data_provider
        self.portfolio = portfolio

    def run(self) -> None:
        """Implement in subclass."""
        raise NotImplementedError


class BasePerformanceMetrics(PerformanceMetrics):
    """Abstract base class for performance metrics."""

    def __init__(self):
        pass

    def calculate(self) -> dict[str, Any]:
        """Implement in subclass."""
        raise NotImplementedError


class BaseConfiguration(Configuration):
    """Abstract base class for configuration."""

    def __init__(self, defaults: dict[str, Any] | None = None):
        self._config = {}
        self._defaults = defaults or {}
        self._validators: dict[str, Any] = {}
        self._errors: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []

        for key, validator in self._validators.items():
            if key in self._config:
                try:
                    if not validator(self._config[key]):
                        errors.append(f"Invalid {key}: {self._config[key]}")
                except Exception as e:
                    errors.append(f"Error validating {key}: {e}")

        self._errors = errors
        return errors

    def add_validator(self, key: str, validator) -> None:
        """Add validator for configuration key."""
        self._validators[key] = validator

    def get_errors(self) -> list[str]:
        """Get validation errors."""
        return self._errors


class BasePlugin(Plugin):
    """Abstract base class for plugins."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._initialized = False

    @property
    def name(self) -> str:
        """Plugin name."""
        return self._name

    @property
    def version(self) -> str:
        """Plugin version."""
        return self._version

    def initialize(self) -> None:
        """Initialize plugin."""
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized


# Common validation functions
def validate_positive_number(value: Any, _name: str) -> bool:
    """Validate positive number."""
    try:
        return Decimal(str(value)) > 0
    except (ValueError, TypeError):
        return False


def validate_percentage(value: Any, _name: str) -> bool:
    """Validate percentage (0-100)."""
    try:
        val = Decimal(str(value))
        return Decimal("0") <= val <= Decimal("100")
    except (ValueError, TypeError):
        return False


def validate_datetime(value: Any, _name: str) -> bool:
    """Validate datetime."""
    return isinstance(value, datetime)


def validate_non_empty_string(value: Any, _name: str) -> bool:
    """Validate non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def validate_symbol(value: Any, _name: str) -> bool:
    """Validate trading symbol."""
    return isinstance(value, str) and len(value.strip()) > 0
