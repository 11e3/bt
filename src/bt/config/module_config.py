"""Configuration management for the backtesting framework."""

from decimal import Decimal
from typing import Any

from bt.domain.types import (
    Amount,
    Fee,
    Percentage,
)
from bt.interfaces.core import (
    Configuration,
)


class ModuleConfiguration(Configuration):
    """Configuration for a specific module."""

    def __init__(self, module_name: str, defaults: dict[str, Any] | None = None):
        super().__init__()
        self.module_name = module_name
        self._config_prefix = module_name.upper() + "_"

        # Initialize configuration
        self._config: dict[str, Any] = defaults or {}

    def _validate_positive_integer(self, value: Any) -> bool:
        """Validate positive integer."""
        try:
            return int(str(value)) > 0
        except (ValueError, TypeError):
            return False

    def _validate_positive_number(self, value: Any) -> bool:
        """Validate positive number."""
        try:
            return Decimal(str(value)) > 0
        except (ValueError, TypeError):
            return False

    def _validate_percentage(self, value: Any) -> bool:
        """Validate percentage (0-100)."""
        try:
            val = Decimal(str(value))
            return Decimal("0") <= val <= Decimal("100")
        except (ValueError, TypeError):
            return False

    def _validate_non_empty_string(self, value: Any) -> bool:
        """Validate non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    def _validate_non_empty_string_list(self, value: Any) -> bool:
        """Validate non-empty list of strings."""
        if not isinstance(value, list):
            return False

        return all(isinstance(v, str) and len(v.strip()) > 0 for v in value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with module prefix."""
        full_key = f"{self._config_prefix}{key}"
        if hasattr(self, "_config") and self._config:
            return self._config.get(full_key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        """Set module-specific configuration."""
        full_key = f"{self._config_prefix}{key}"
        self._config[full_key] = value

    def get_all(self) -> dict[str, Any]:
        """Get all configuration for this module."""
        result = {}
        if hasattr(self, "_config"):
            for key in self._config:
                if key.startswith(self._config_prefix):
                    result[key] = self._config[key]
        return result


class BacktestConfiguration(ModuleConfiguration):
    """Configuration for backtesting."""

    def __init__(self, defaults: dict[str, Any] | None = None):
        # Backtest-specific defaults
        backtest_defaults = {
            "initial_cash": 10000000,
            "fee": 0.001,
            "slippage": 0.001,
            "lookback": 5,
            "multiplier": 2,
            "interval": "day",
            "symbols": [],
        }

        if defaults:
            backtest_defaults.update(defaults)

        super().__init__("BACKTEST_", backtest_defaults)

    def get_initial_cash(self) -> Amount:
        """Get initial cash as Amount."""
        return Amount(Decimal(str(self.get("initial_cash", 10000000))))

    def get_fee(self) -> Fee:
        """Get fee as Fee."""
        return Fee(Decimal(str(self.get("fee", "0.001"))))

    def get_slippage(self) -> Percentage:
        """Get slippage as Percentage."""
        return Percentage(Decimal(str(self.get("slippage", "0.001"))))

    def get_symbols(self) -> list[str]:
        """Get symbols as list."""
        symbols = self.get("symbols", [])
        if isinstance(symbols, str):
            return [symbols]  # Convert single string to list
        return symbols

    def get_lookback(self) -> int:
        """Get lookback period."""
        return int(self.get("lookback", 5))

    def get_multiplier(self) -> int:
        """Get lookback multiplier."""
        return int(self.get("multiplier", 2))

    def get_mom_lookback(self) -> int:
        """Get momentum lookback."""
        return int(self.get("mom_lookback", 20))

    def get_top_n(self) -> int:
        """Get top N symbols."""
        return int(self.get("top_n", 3))

    def get_interval(self) -> str:
        """Get data interval."""
        return self.get("interval", "day")


# Global configuration registry
_config_registry: dict[str, ModuleConfiguration] = {}


def register_config(module_config: ModuleConfiguration) -> None:
    """Register a module configuration."""
    _config_registry[module_config.module_name] = module_config


def get_config(module_name: str) -> ModuleConfiguration | None:
    """Get configuration for a module."""
    return _config_registry.get(module_name)
