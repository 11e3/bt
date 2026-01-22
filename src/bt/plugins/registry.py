"""Plugin registry for managing different types of plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bt.exceptions import StrategyError
from bt.plugins.interfaces import DataProviderPlugin, ReporterPlugin, StrategyPlugin
from bt.plugins.manager import PluginManager
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class PluginRegistry:
    """Registry for managing different types of plugins."""

    def __init__(self) -> None:
        self._strategies: dict[str, type] = {}
        self._data_providers: dict[str, type] = {}
        self._reporters: dict[str, type] = {}
        self._manager = PluginManager()

    def register_strategy_plugin(self, plugin: StrategyPlugin) -> None:
        """Register a strategy plugin."""
        strategy_class = plugin.get_strategy_class()
        strategy_name = getattr(strategy_class, "get_name", lambda: plugin.name)()

        self._strategies[strategy_name] = strategy_class
        logger.info(f"Registered strategy plugin: {strategy_name}")

    def register_data_provider_plugin(self, plugin: DataProviderPlugin) -> None:
        """Register a data provider plugin."""
        provider_class = plugin.get_data_provider_class()
        provider_name = plugin.name

        self._data_providers[provider_name] = provider_class
        logger.info(f"Registered data provider plugin: {provider_name}")

    def register_reporter_plugin(self, plugin: ReporterPlugin) -> None:
        """Register a reporter plugin."""
        reporter_class = plugin.get_reporter_class()
        reporter_name = plugin.name

        self._reporters[reporter_name] = reporter_class
        logger.info(f"Registered reporter plugin: {reporter_name}")

    def get_strategy_class(self, name: str) -> type:
        """Get a strategy class by name."""
        if name not in self._strategies:
            raise StrategyError(f"Strategy plugin '{name}' not found")
        return self._strategies[name]

    def get_data_provider_class(self, name: str) -> type:
        """Get a data provider class by name."""
        if name not in self._data_providers:
            raise ValueError(f"Data provider plugin '{name}' not found")
        return self._data_providers[name]

    def get_reporter_class(self, name: str) -> type:
        """Get a reporter class by name."""
        if name not in self._reporters:
            raise ValueError(f"Reporter plugin '{name}' not found")
        return self._reporters[name]

    def list_strategies(self) -> list[str]:
        """List all registered strategy plugins."""
        return list(self._strategies.keys())

    def list_data_providers(self) -> list[str]:
        """List all registered data provider plugins."""
        return list(self._data_providers.keys())

    def list_reporters(self) -> list[str]:
        """List all registered reporter plugins."""
        return list(self._reporters.keys())

    def discover_and_load_plugins(self, plugin_paths: list[Path] | None = None) -> None:
        """Discover and load all available plugins."""
        if plugin_paths:
            for path in plugin_paths:
                self._manager.register_plugin_path(path)

        discovered = self._manager.discover_plugins()

        for plugin_name in discovered:
            plugin = self._manager.load_plugin(plugin_name)

            # Register with appropriate registry based on type
            if isinstance(plugin, StrategyPlugin):
                self.register_strategy_plugin(plugin)
            elif isinstance(plugin, DataProviderPlugin):
                self.register_data_provider_plugin(plugin)
            elif isinstance(plugin, ReporterPlugin):
                self.register_reporter_plugin(plugin)
