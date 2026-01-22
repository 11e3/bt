"""Plugin interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PluginInterface(ABC):
    """Base interface for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        pass


class StrategyPlugin(PluginInterface):
    """Plugin interface for trading strategies."""

    @abstractmethod
    def get_strategy_class(self) -> type:
        """Return the strategy class to register."""
        pass

    @abstractmethod
    def get_strategy_config_schema(self) -> dict[str, Any]:
        """Return JSON schema for strategy configuration."""
        pass


class DataProviderPlugin(PluginInterface):
    """Plugin interface for data providers."""

    @abstractmethod
    def get_data_provider_class(self) -> type:
        """Return the data provider class to register."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> list[str]:
        """Return list of supported symbols."""
        pass


class ReporterPlugin(PluginInterface):
    """Plugin interface for reporting components."""

    @abstractmethod
    def get_reporter_class(self) -> type:
        """Return the reporter class to register."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        pass
