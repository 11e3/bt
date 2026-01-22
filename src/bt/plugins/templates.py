"""Plugin templates and utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pkg_resources

from bt.plugins.manager import PluginManager
from bt.plugins.registry import PluginRegistry
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.plugins.interfaces import PluginInterface

logger = get_logger(__name__)

# Global plugin registry instance
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _plugin_registry


def set_plugin_registry(registry: PluginRegistry) -> None:
    """Set the global plugin registry."""
    global _plugin_registry
    _plugin_registry = registry


def load_plugin_from_entry_point(group: str, name: str) -> PluginInterface | None:
    """Load a plugin from setuptools entry points."""
    try:
        for entry_point in pkg_resources.iter_entry_points(group, name):
            plugin_class = entry_point.load()
            return plugin_class()
    except Exception as e:
        logger.error(f"Failed to load plugin '{name}' from entry points: {e}")

    return None


def load_plugin_from_file(filepath: str | Path) -> PluginInterface | None:
    """Load a plugin from a Python file."""
    try:
        plugin_path = Path(filepath)
        plugin_manager = PluginManager()
        plugin_manager.register_plugin_path(plugin_path.parent)

        # Extract plugin name from filename
        plugin_name = plugin_path.stem
        return plugin_manager.load_plugin(plugin_name)

    except Exception as e:
        logger.error(f"Failed to load plugin from file '{filepath}': {e}")

    return None


def create_plugin_template(plugin_type: str, name: str, output_dir: str | Path = ".") -> Path:
    """Create a template plugin file."""

    templates = {
        "strategy": f'''"""{name} trading strategy plugin."""

from bt.plugins import StrategyPlugin
from bt.strategies.implementations import BaseStrategy


class {name}Strategy(BaseStrategy):
    """Custom {name} trading strategy."""

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors = []
        # Add validation logic here
        return errors

    def get_buy_conditions(self) -> dict:
        """Define buy conditions."""
        return {{
            "custom_condition": lambda engine, symbol: True  # Replace with actual logic
        }}

    def get_allocation_func(self) -> callable:
        """Define position sizing."""
        return lambda engine, symbol, price: 1000  # Replace with actual logic


class {name}StrategyPlugin(StrategyPlugin):
    """Plugin wrapper for {name} strategy."""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_strategy_class(self):
        return {name}Strategy

    def get_strategy_config_schema(self) -> dict:
        return {{
            "type": "object",
            "properties": {{
                "param1": {{"type": "number", "default": 1.0}}
            }}
        }}

    def initialize(self, config: dict) -> None:
        """Initialize the plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
''',
        "data_provider": f'''"""{name} data provider plugin."""

from bt.plugins import DataProviderPlugin
from bt.interfaces.protocols import IDataProvider


class {name}DataProvider(IDataProvider):
    """Custom {name} data provider."""

    def __init__(self):
        self._data = {{}}

    @property
    def symbols(self) -> list[str]:
        return list(self._data.keys())

    def load_data(self, symbol: str, df) -> None:
        self._data[symbol] = df

    def get_bar(self, symbol: str, offset: int = 0):
        if symbol in self._data:
            df = self._data[symbol]
            idx = len(df) - 1 - offset
            if idx >= 0:
                return df.iloc[idx]
        return None

    def get_bars(self, symbol: str, count: int):
        if symbol in self._data:
            return self._data[symbol].tail(count)
        return None

    def has_more_data(self) -> bool:
        return any(len(df) > 0 for df in self._data.values())

    def next_bar(self) -> None:
        # Implementation for data iteration
        pass

    def set_current_bar(self, symbol: str, index: int) -> None:
        # Implementation for bar positioning
        pass

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        return {{symbol: 100.0 for symbol in symbols}}  # Replace with actual logic

    def get_current_datetime_batch(self, symbols: list[str]):
        from datetime import datetime
        return datetime.now()


class {name}DataProviderPlugin(DataProviderPlugin):
    """Plugin wrapper for {name} data provider."""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_data_provider_class(self):
        return {name}DataProvider

    def get_supported_symbols(self) -> list[str]:
        return ["SYMBOL1", "SYMBOL2"]  # Replace with actual symbols

    def initialize(self, config: dict) -> None:
        """Initialize the plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
''',
    }

    if plugin_type not in templates:
        raise ValueError(f"Unknown plugin type: {plugin_type}")

    output_path = Path(output_dir) / f"{name}_plugin.py"
    output_path.write_text(templates[plugin_type])

    return output_path
